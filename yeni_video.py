import cv2
import numpy as np

def process_frame(frame):
    # 1. Gri Tonlama
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Kenar Tespiti
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. ROI Maskesi
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Buradaki vertices, ROI 
    vertices = np.array([[
        (width * 0.25, height * 0.972),   # Sol-alt
        (width * 0.75, height * 0.972),   # Sağ-alt
        (width * 0.55, height * 0.78),    # Sağ-üst
        (width * 0.40, height * 0.78)     # Sol-üst
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 5. Hough Çizgileri
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=15,
        minLineLength=40,     # daha kısa parçaları da al
        maxLineGap=50         # daha büyük boşluklara izin ver
    )
    
    # Çizgileri çiz
    line_image = np.zeros_like(frame)
    if lines is not None:
        # 5a. Çizgileri sol/sağ olarak ayır
        left_lines, right_lines = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))
        
        # 5b. Her gruptan ortalama doğru hesapla
        def avg_line(lines):
            slopes, intercepts = [], []
            for x1, y1, x2, y2 in lines:
                m, b = np.polyfit((x1, x2), (y1, y2), 1)
                slopes.append(m)
                intercepts.append(b)
            return np.mean(slopes), np.mean(intercepts)
        
        left_m, left_b   = avg_line(left_lines) if left_lines else (0,0)
        right_m, right_b = avg_line(right_lines) if right_lines else (0,0)
        
        # 5c. Ortalama doğruları görüntünün alt ve üst sınırına uzat
        y_bottom = height
        y_top    = int(height * 0.6)
        def line_points(m, b, y1, y2):
            x1 = int((y1 - b) / (m + 1e-6))
            x2 = int((y2 - b) / (m + 1e-6))
            return (x1, y1), (x2, y2)
        
        lp1, lp2 = line_points(left_m, left_b, y_bottom, y_top)
        rp1, rp2 = line_points(right_m, right_b, y_bottom, y_top)
        
        # 6. Şerit alanını dolduracak poligon
        fill_poly = np.array([[lp1, lp2, rp2, rp1]], dtype=np.int32)
        
        # 7. Dolgu uygulaması (yarı saydam)
        overlay = frame.copy()
        cv2.fillPoly(overlay, fill_poly, color=(255, 0, 0))  # BGR mavi
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # 8. Hough’dan dönen tüm çizgileri üzerine çiz
        for x1, y1, x2, y2 in lines.reshape(-1,4):
            cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 3)
    else:
        # Eğer hiç çizgi yoksa, yine de boş line_image ile devam et
        pass
    
    # —————————————————————————————————————————————————————————
    # **EKLENECEK KISIM**: ROI köşe noktalarını işaretle
    for pt in vertices[0]:
        cv2.circle(
            frame,
            center=(int(pt[0]), int(pt[1])),
            radius=6,
            color=(0, 0, 255),    # Kırmızı nokta
            thickness=-1          # dolu daire
        )
    # —————————————————————————————————————————————————————————
    
    # Sonucu birleştir ve döndür
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

# Video Kaynağı (Dosya yolu veya 0 for webcam)
video_path = "Araba.mp4"  # Kendi videonuzun yolunu verin
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    cv2.imshow("Lane Detection & ROI Points", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
