using System;
using System.Threading.Tasks;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

public class PoseDetection : MonoBehaviour
{
    public PosePreview posePreview;
    public ImagePreview imagePreview;
    public Texture2D imageTexture;
    public bool useWebcamTexture = true;
    public InferenceWebcam webcamSource;
    public bool waitForInferenceWebcamTrigger = true;
    [Min(1)] public int detectEveryNthWebcamFrame = 4;
    public ModelAsset poseDetector;
    public ModelAsset poseLandmarker;
    public TextAsset anchorsCSV;

    public float scoreThreshold = 0.75f;

    const int k_NumAnchors = 2254;
    float[,] m_Anchors;

    const int k_NumKeypoints = 33;
    const int k_LeftShoulderIndex = 11;
    const int k_RightShoulderIndex = 12;
    const int k_LeftWristIndex = 15;
    const int k_RightWristIndex = 16;
    const int detectorInputSize = 224;
    const int landmarkerInputSize = 256;

    Worker m_PoseDetectorWorker;
    Worker m_PoseLandmarkerWorker;
    Tensor<float> m_DetectorInput;
    Tensor<float> m_LandmarkerInput;
    Awaitable m_DetectAwaitable;
    WebCamTexture m_WebCamTexture;
    Texture m_InputTexture;
    int m_WebcamFrameCount;
    bool m_ShuttingDown;
    bool m_InitializationStarted;
    bool m_ClapArmed = true;
    float m_LastClapTime = -10f;
    Coroutine m_ClapFlashCoroutine;

    [Header("Clap Detection")]
    public bool enableClapDetection = true;
    public RawImage clapFlashRawImage;
    [Min(0.01f)] public float clapFlashFadeSeconds = 0.35f;
    [Tooltip("Wrists must be closer than this factor * shoulder width to count as a clap.")]
    [Range(0.1f, 2f)] public float clapDistanceFactor = 0.35f;
    [Tooltip("Minimum time between clap events in seconds.")]
    [Range(0.05f, 2f)] public float clapCooldownSeconds = 0.3f;

    float m_TextureWidth;
    float m_TextureHeight;

    void Start()
    {
        if (useWebcamTexture && waitForInferenceWebcamTrigger)
            return;

        BeginDetection();
    }

    public void StartFromInferenceWebcam(InferenceWebcam source)
    {
        webcamSource = source;
        BeginDetection();
    }

    public void SetClapSensitivity(float value)
    {
        clapDistanceFactor = Mathf.Clamp(value, 0.1f, 2f);
    }

    async void BeginDetection()
    {
        if (m_InitializationStarted)
            return;
        m_InitializationStarted = true;

        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        var poseDetectorModel = ModelLoader.Load(poseDetector);
        // post process the model to filter scores + argmax select the best pose
        var graph = new FunctionalGraph();
        var input = graph.AddInput(poseDetectorModel, 0);
        var outputs = Functional.Forward(poseDetectorModel, input);
        var boxes = outputs[0]; // (1, 2254, 12)
        var scores = outputs[1]; // (1, 2254, 1)
        var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
        poseDetectorModel = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        m_PoseDetectorWorker = new Worker(poseDetectorModel, BackendType.GPUCompute);

        var poseLandmarkerModel = ModelLoader.Load(poseLandmarker);
        m_PoseLandmarkerWorker = new Worker(poseLandmarkerModel, BackendType.GPUCompute);

        m_DetectorInput = new Tensor<float>(new TensorShape(1, detectorInputSize, detectorInputSize, 3));
        m_LandmarkerInput = new Tensor<float>(new TensorShape(1, landmarkerInputSize, landmarkerInputSize, 3));

        if (useWebcamTexture)
        {
            if (webcamSource == null)
                webcamSource = FindObjectOfType<InferenceWebcam>();

            if (webcamSource != null && webcamSource.WebcamTexture != null)
            {
                m_WebCamTexture = webcamSource.WebcamTexture;
                m_InputTexture = m_WebCamTexture;
            }
            else
            {
                Debug.LogWarning("InferenceWebcam source/webcam is not available. Falling back to imageTexture.");
                m_InputTexture = imageTexture;
            }
        }
        else
        {
            m_InputTexture = imageTexture;
        }

        if (m_InputTexture == null)
        {
            Debug.LogError("No input texture available for pose detection.");
            return;
        }

        while (true)
        {
            try
            {
                if (m_ShuttingDown)
                    break;

                if (m_WebCamTexture != null)
                {
                    // Wait for a valid camera frame before running inference.
                    if (!m_WebCamTexture.didUpdateThisFrame || m_WebCamTexture.width <= 16 || m_WebCamTexture.height <= 16)
                    {
                        await Task.Yield();
                        continue;
                    }

                    m_WebcamFrameCount++;
                    if (m_WebcamFrameCount % Mathf.Max(1, detectEveryNthWebcamFrame) != 0)
                    {
                        await Task.Yield();
                        continue;
                    }
                }

                m_DetectAwaitable = Detect(m_InputTexture);
                await m_DetectAwaitable;
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        m_PoseDetectorWorker.Dispose();
        m_PoseLandmarkerWorker.Dispose();
        m_DetectorInput.Dispose();
        m_LandmarkerInput.Dispose();
    }

    Vector3 ImageToWorld(Vector2 position)
    {
        return (position - 0.5f * new Vector2(m_TextureWidth, m_TextureHeight)) / m_TextureHeight;
    }

    async Awaitable Detect(Texture texture)
    {
        m_TextureWidth = texture.width;
        m_TextureHeight = texture.height;
        // imagePreview.SetTexture(texture);

        var size = Mathf.Max(texture.width, texture.height);

        // The affine transformation matrix to go from tensor coordinates to image coordinates
        var scale = size / (float)detectorInputSize;
        var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

        m_PoseDetectorWorker.Schedule(m_DetectorInput);

        var outputIdxAwaitable = (m_PoseDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoreAwaitable = (m_PoseDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxAwaitable = (m_PoseDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIdx = await outputIdxAwaitable;
        using var outputScore = await outputScoreAwaitable;
        using var outputBox = await outputBoxAwaitable;

        var scorePassesThreshold = outputScore[0] >= scoreThreshold;
        posePreview.SetActive(scorePassesThreshold);

        if (!scorePassesThreshold)
            return;

        var idx = outputIdx[0];

        var anchorPosition = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);

        var face_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 0], outputBox[0, 0, 1]));
        var faceTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 0] + 0.5f * outputBox[0, 0, 2], outputBox[0, 0, 1] + 0.5f * outputBox[0, 0, 3]));

        var kp1_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 0 + 0], outputBox[0, 0, 4 + 2 * 0 + 1]));
        var kp2_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 1 + 0], outputBox[0, 0, 4 + 2 * 1 + 1]));
        var delta_ImageSpace = kp2_ImageSpace - kp1_ImageSpace;
        var dscale = 1.25f;
        var radius = dscale * math.length(delta_ImageSpace);
        var theta = math.atan2(delta_ImageSpace.y, delta_ImageSpace.x);
        var origin2 = new float2(0.5f * landmarkerInputSize, 0.5f * landmarkerInputSize);
        var scale2 = radius / (0.5f * landmarkerInputSize);
        var M2 = BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(kp1_ImageSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(0.5f * Mathf.PI - theta)), BlazeUtils.TranslationMatrix(-origin2));
        BlazeUtils.SampleImageAffine(texture, m_LandmarkerInput, M2);

        var boxSize = 2f * (faceTopRight_ImageSpace - face_ImageSpace);

        posePreview.SetBoundingBox(true, ImageToWorld(face_ImageSpace), boxSize / m_TextureHeight);
        posePreview.SetBoundingCircle(true, ImageToWorld(kp1_ImageSpace), radius / m_TextureHeight);

        m_PoseLandmarkerWorker.Schedule(m_LandmarkerInput);

        var landmarksAwaitable = (m_PoseLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
        using var landmarks = await landmarksAwaitable; // (1,195)

        for (var i = 0; i < k_NumKeypoints; i++)
        {
            // https://arxiv.org/pdf/2006.10204
            var position_ImageSpace = BlazeUtils.mul(M2, new float2(landmarks[5 * i + 0], landmarks[5 * i + 1]));
            var visibility = landmarks[5 * i + 3];
            var presence = landmarks[5 * i + 4];

            // z-position is in unit cube centered on hips
            Vector3 position_WorldSpace = ImageToWorld(position_ImageSpace) + new Vector3(0, 0, landmarks[5 * i + 2] / m_TextureHeight);
            posePreview.SetKeypoint(i, visibility > 0.5f && presence > 0.5f, position_WorldSpace);
        }

        if (enableClapDetection)
            DetectClap(landmarks, M2);
    }

    void DetectClap(Tensor<float> landmarks, float2x3 M2)
    {
        bool IsVisible(int keypointIndex)
        {
            var visibility = landmarks[5 * keypointIndex + 3];
            var presence = landmarks[5 * keypointIndex + 4];
            return visibility > 0.5f && presence > 0.5f;
        }

        Vector2 GetImageSpacePosition(int keypointIndex)
        {
            return BlazeUtils.mul(M2, new float2(landmarks[5 * keypointIndex + 0], landmarks[5 * keypointIndex + 1]));
        }

        var wristsVisible = IsVisible(k_LeftWristIndex) && IsVisible(k_RightWristIndex);
        var shouldersVisible = IsVisible(k_LeftShoulderIndex) && IsVisible(k_RightShoulderIndex);

        if (!wristsVisible || !shouldersVisible)
        {
            m_ClapArmed = true;
            return;
        }

        var leftWrist = GetImageSpacePosition(k_LeftWristIndex);
        var rightWrist = GetImageSpacePosition(k_RightWristIndex);
        var leftShoulder = GetImageSpacePosition(k_LeftShoulderIndex);
        var rightShoulder = GetImageSpacePosition(k_RightShoulderIndex);

        var shoulderWidth = Vector2.Distance(leftShoulder, rightShoulder);
        if (shoulderWidth <= 0.0001f)
            return;

        var wristDistance = Vector2.Distance(leftWrist, rightWrist);
        var clapThreshold = shoulderWidth * clapDistanceFactor;
        var isClapPose = wristDistance <= clapThreshold;

        if (isClapPose && m_ClapArmed && Time.time - m_LastClapTime >= clapCooldownSeconds)
        {
            Debug.Log("Clap detected");
            m_LastClapTime = Time.time;
            m_ClapArmed = false;
            TriggerClapFlash();
            return;
        }

        // Rearm once hands separate so one clap only logs once.
        if (!isClapPose)
            m_ClapArmed = true;
    }

    void TriggerClapFlash()
    {
        if (clapFlashRawImage == null)
            return;

        if (m_ClapFlashCoroutine != null)
            StopCoroutine(m_ClapFlashCoroutine);

        m_ClapFlashCoroutine = StartCoroutine(ClapFlashRoutine());
    }

    System.Collections.IEnumerator ClapFlashRoutine()
    {
        var color = clapFlashRawImage.color;
        color.a = 1f;
        clapFlashRawImage.color = color;

        var duration = Mathf.Max(0.01f, clapFlashFadeSeconds);
        var elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            var t = Mathf.Clamp01(elapsed / duration);
            color.a = Mathf.Lerp(1f, 0f, t);
            clapFlashRawImage.color = color;
            yield return null;
        }

        color.a = 0f;
        clapFlashRawImage.color = color;
        m_ClapFlashCoroutine = null;
    }

    void OnDestroy()
    {
        m_ShuttingDown = true;

        if (m_ClapFlashCoroutine != null)
            StopCoroutine(m_ClapFlashCoroutine);
        m_DetectAwaitable.Cancel();
    }
}
