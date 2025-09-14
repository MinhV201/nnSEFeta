import torch
from torchvision import transforms
from PIL import Image
import cv2

# ===== 1. Khai báo tên class trực tiếp =====
classes_class = ['AC', 'FL', 'Head']        
classes_qualify = ['Average', 'High', 'Low'] 

# ===== 2. Load model =====
from model_architech import FETA  

model = FETA(input_channels=3, out_channels=3)
model.load_state_dict(torch.load("/home/vuminh/nn_UNetFrame/nnSEFeta/model1.pth", map_location="cpu"))
model.eval()

# ===== 3. Transform  =====
transform = transforms.Compose([
    transforms.CenterCrop(640),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== 4. Hàm dự đoán 1 frame =====
def predict_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(input_tensor)

    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        class_pred, qualify_pred = outputs
    else:
        raise ValueError("Model cần trả về tuple (class_pred, qualify_pred)")

    class_idx = torch.argmax(class_pred, dim=1).item()
    qualify_idx = torch.argmax(qualify_pred, dim=1).item()

    # Đổi thành tên
    class_name = classes_class[class_idx]
    qualify_name = classes_qualify[qualify_idx]
    print(f"{class_name} | {qualify_name}")
    return class_name, qualify_name
    

# ===== 5. Chạy realtime =====
def run_realtime(video_path: str):

    cap = cv2.VideoCapture(video_path)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_name, qualify_name = predict_frame(frame)
        if class_name is not None:
            text = f"Class: {class_name} | Qualify: {qualify_name}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("Realtime Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime(video_path="/home/vuminh/nn_UNetFrame/nnSEFeta/video_all.mp4")
