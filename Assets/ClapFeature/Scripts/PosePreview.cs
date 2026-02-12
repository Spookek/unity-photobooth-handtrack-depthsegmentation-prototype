using UnityEngine;

public class PosePreview : MonoBehaviour
{
    public BoundingBox boundingBox;
    public BoundingCircle boundingCircle;
    public Keypoint[] keypoints;

    public void SetActive(bool active)
    {
        gameObject.SetActive(active);
    }

    public void SetBoundingBox(bool active, Vector3 position, Vector2 size)
    {
        boundingBox.Set(active, position, size);
    }

    public void SetBoundingCircle(bool active, Vector3 position, float radius)
    {
        boundingCircle.Set(active, position, radius);
    }

    public void SetKeypoint(int index, bool active, Vector3 position)
    {
        keypoints[index].Set(active, position);
    }

    public void SetAllKeypointsActive(bool active)
    {
        for (var i = 0; i < keypoints.Length; i++)
            keypoints[i].Set(active, keypoints[i].Position);
    }
}
