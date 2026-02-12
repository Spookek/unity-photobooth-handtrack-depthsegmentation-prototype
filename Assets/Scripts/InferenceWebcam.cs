using System.Collections;
using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;

public class InferenceWebcam : MonoBehaviour
{
    public ModelAsset estimationModel;
    public PoseDetection poseDetection;
    Worker m_engineEstimation;
    public WebCamTexture webcamTexture;
    public RawImage rawImage;
    Tensor<float> inputTensor;
    RenderTexture outputTexture;

    public Material material;
    public Texture2D colorMap;
    public Texture backgroundImage;
    public Texture foregroundImage;
    public Slider depthMinSlider;
    public Slider debugViewSlider;
    public Slider clapSensitivitySlider;
    public Toggle foregroundToggle;
    Texture2D transparentFallbackTexture;
    bool m_ShowForeground = true;

    int modelLayerCount = 0;
    public int framesToExectute = 2;
    public WebCamTexture WebcamTexture => webcamTexture;

    void Start()
    {
        Application.targetFrameRate = 60;
        var model = ModelLoader.Load(estimationModel);

        // Post process
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var outputs = Functional.Forward(model, inputs);
        var output = outputs[0];

        var max0 = Functional.ReduceMax(output, new[] { 1, 2 }, false);
        var min0 = Functional.ReduceMin(output, new[] { 1, 2 }, false);
        output = (output - min0) / (max0 - min0);

        model = graph.Compile(output);
        modelLayerCount = model.layers.Count;

        m_engineEstimation = new Worker(model, BackendType.GPUCompute);

        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("No webcam devices found.");
            return;
        }

        webcamTexture = new WebCamTexture(1920, 1080);
        webcamTexture.deviceName = devices[0].name;
        webcamTexture.Play();
        StartCoroutine(StartPoseDetectionWhenWebcamReady());

        outputTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBFloat);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 256, 256));
        InitializeSliderBindings();
    }

    void InitializeSliderBindings()
    {
        if (depthMinSlider != null)
        {
            depthMinSlider.onValueChanged.AddListener(OnDepthMinSliderChanged);
            if (material != null)
                depthMinSlider.value = material.GetFloat("_DepthMin");
        }

        if (debugViewSlider != null)
        {
            debugViewSlider.onValueChanged.AddListener(OnDebugViewSliderChanged);
            if (material != null)
                debugViewSlider.value = material.GetFloat("_DebugView");
        }

        if (clapSensitivitySlider != null)
        {
            clapSensitivitySlider.onValueChanged.AddListener(OnClapSensitivitySliderChanged);
            if (poseDetection != null)
                clapSensitivitySlider.value = poseDetection.clapDistanceFactor;
        }

        if (foregroundToggle != null)
        {
            foregroundToggle.onValueChanged.AddListener(OnForegroundToggleChanged);
            foregroundToggle.SetIsOnWithoutNotify(true);
            OnForegroundToggleChanged(true);
        }
        else
        {
            m_ShowForeground = true;
            if (material != null)
                material.SetFloat("_ForegroundOpacity", 1f);
        }
    }

    public void OnDepthMinSliderChanged(float value)
    {
        if (material == null)
            return;

        material.SetFloat("_DepthMin", value);
    }

    public void OnClapSensitivitySliderChanged(float value)
    {
        if (poseDetection == null)
            poseDetection = FindObjectOfType<PoseDetection>();

        if (poseDetection == null)
            return;

        poseDetection.SetClapSensitivity(value);
    }

    public void OnDebugViewSliderChanged(float value)
    {
        if (material == null)
            return;

        material.SetFloat("_DebugView", value);
    }

    public void OnForegroundToggleChanged(bool isOn)
    {
        m_ShowForeground = isOn;

        if (material == null)
            return;

        material.SetFloat("_ForegroundOpacity", m_ShowForeground ? 1f : 0f);
    }

    IEnumerator StartPoseDetectionWhenWebcamReady()
    {
        if (poseDetection == null)
            poseDetection = FindObjectOfType<PoseDetection>();

        if (poseDetection == null)
            yield break;

        while (webcamTexture == null || webcamTexture.width <= 16 || webcamTexture.height <= 16 || !webcamTexture.didUpdateThisFrame)
            yield return null;

        poseDetection.StartFromInferenceWebcam(this);
    }

    bool executionStarted = false;
    IEnumerator executionSchedule;
    void Update()
    {
        if (webcamTexture == null || m_engineEstimation == null || inputTensor == null)
            return;

        if (!executionStarted)
        {
            TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());
            executionSchedule = m_engineEstimation.ScheduleIterable(inputTensor);
            executionStarted = true;
        }

        bool hasMoreWork = false;
        int layersToRun = (modelLayerCount + framesToExectute - 1) / framesToExectute; // round up
        for (int i = 0; i < layersToRun; i++)
        {
            hasMoreWork = executionSchedule.MoveNext();
            if (!hasMoreWork)
                break;
        }

        if (hasMoreWork)
            return;

        var output = m_engineEstimation.PeekOutput() as Tensor<float>;
        output.Reshape(output.shape.Unsqueeze(0));
        TextureConverter.RenderToTexture(output, outputTexture, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));
        executionStarted = false;
    }

    void OnRenderObject()
    {
        if (material == null || webcamTexture == null || outputTexture == null)
            return;

        material.SetVector("ScreenCamResolution", new Vector4(Screen.height, Screen.width, 0, 0));
        material.SetTexture("WebCamTex", webcamTexture);
        material.SetTexture("DepthTex", outputTexture);
        material.SetTexture("ColorRampTex", colorMap);
        material.SetTexture("BackgroundTex", backgroundImage != null ? backgroundImage : Texture2D.blackTexture);
        material.SetTexture("ForegroundTex", foregroundImage != null ? foregroundImage : GetTransparentFallbackTexture());
        material.SetFloat("_ForegroundOpacity", m_ShowForeground ? 1f : 0f);
        if (rawImage != null)
        {
            material.SetFloat("_UseMeshUV", 1f);
            rawImage.material = material;
            if (rawImage.texture == null)
                rawImage.texture = Texture2D.whiteTexture;
            return;
        }

        material.SetFloat("_UseMeshUV", 0f);
        Graphics.Blit(null, material);
    }

    Texture2D GetTransparentFallbackTexture()
    {
        if (transparentFallbackTexture != null)
            return transparentFallbackTexture;

        transparentFallbackTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        transparentFallbackTexture.SetPixel(0, 0, Color.clear);
        transparentFallbackTexture.Apply(false, true);
        return transparentFallbackTexture;
    }

    void OnDestroy()
    {
        if (depthMinSlider != null)
            depthMinSlider.onValueChanged.RemoveListener(OnDepthMinSliderChanged);
        if (debugViewSlider != null)
            debugViewSlider.onValueChanged.RemoveListener(OnDebugViewSliderChanged);
        if (clapSensitivitySlider != null)
            clapSensitivitySlider.onValueChanged.RemoveListener(OnClapSensitivitySliderChanged);
        if (foregroundToggle != null)
            foregroundToggle.onValueChanged.RemoveListener(OnForegroundToggleChanged);
        if (m_engineEstimation != null)
            m_engineEstimation.Dispose();
        if (inputTensor != null)
            inputTensor.Dispose();
        if (outputTexture != null)
            outputTexture.Release();
        if (webcamTexture != null && webcamTexture.isPlaying)
            webcamTexture.Stop();
        if (transparentFallbackTexture != null)
            Destroy(transparentFallbackTexture);
    }
}
