import cv2
from app.traffic_light_detector import traffic_light_detector

def test_countdown_detection(image_path):
    """Test countdown detection on an image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Simulate a traffic light bounding box (you can adjust this)
    height, width = image.shape[:2]
    bbox = [width//3, height//3, 2*width//3, 2*height//3]
    
    # Analyze traffic light
    result = traffic_light_detector.analyze_traffic_light(image, bbox)
    
    print("=== Traffic Light Analysis ===")
    print(f"Color: {result['color']}")
    print(f"Countdown: {result['countdown'] if result['countdown'] else 'NOT DETECTED'}")
    print(f"Countdown Detected: {result['countdown_detected']}")
    print(f"Confidence: {result['confidence']}")
    
    # Draw results on image
    x1, y1, x2, y2 = result['bbox']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw expanded search area
    ex1, ey1, ex2, ey2 = result['expanded_bbox']
    cv2.rectangle(image, (ex1, ey1), (ex2, ey2), (255, 0, 0), 1)
    
    # Add text
    text = f"Color: {result['color']}"
    if result['countdown']:
        text += f" | Countdown: {result['countdown']}s"
    else:
        text += " | No countdown"
    
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow("Traffic Light Analysis", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with your image
    test_countdown_detection("uploads/1.jpg")