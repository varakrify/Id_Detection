from ultralytics import YOLO

# Load trained model using raw string for Windows path
model = YOLO(r"C:\Users\User\Documents\AIML\Id_Detection\runs\detect\id_card_detector2\weights\best.pt")

# Test on a sample image
results = model(r"C:\Users\User\Documents\AIML\Id_Detection\withId2.jpg")

# Show annotated image
results[0].show()

# Check if ID card is detected
if len(results[0].boxes) > 0:
    print("ID Card Present ✅")
else:
    print("ID Card Not Present ❌")
