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
        (width * 0.54, height * 0.765),    # Sağ-üst
        (width * 0.45, height * 0.765)     # Sol-üst
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 5. Hough Çizgileri - Parametreleri iyileştirildi
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20,        # Eşik değeri artırıldı
        minLineLength=60,    # Minimum çizgi uzunluğu artırıldı
        maxLineGap=30        # Maksimum boşluk azaltıldı
    )
    
    # Çizgileri çiz
    line_image = np.zeros_like(frame)
    if lines is not None:
        # 5a. Çizgileri sol/sağ olarak ayır ve filtrele
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Dikey çizgileri atla
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # Yatay çizgileri atla
                continue
            if slope < -0.3:  # Sol şerit
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:  # Sağ şerit
                right_lines.append((x1, y1, x2, y2))
        
        # 5b. Her gruptan ortalama doğru hesapla
        def avg_line(lines):
            if not lines:
                return None, None
            slopes, intercepts = [], []
            for x1, y1, x2, y2 in lines:
                m, b = np.polyfit((x1, x2), (y1, y2), 1)
                slopes.append(m)
                intercepts.append(b)
            return np.mean(slopes), np.mean(intercepts)
        
        left_m, left_b = avg_line(left_lines)
        right_m, right_b = avg_line(right_lines)
        
        # 5c. Ortalama doğruları görüntünün alt ve üst sınırına uzat
        y_bottom = height
        y_top = int(height * 0.78)
        
        def line_points(m, b, y1, y2):
            if m is None or b is None:
                return None, None
            x1 = int((y1 - b) / (m + 1e-6))
            x2 = int((y2 - b) / (m + 1e-6))
            return (x1, y1), (x2, y2)
        
        lp1, lp2 = line_points(left_m, left_b, y_bottom, y_top)
        rp1, rp2 = line_points(right_m, right_b, y_bottom, y_top)
        
        # 6. Şerit alanını dolduracak poligon
        if lp1 and lp2 and rp1 and rp2:
            fill_poly = np.array([[lp1, lp2, rp2, rp1]], dtype=np.int32)
            
            # 7. Dolgu uygulaması (yarı saydam)
            overlay = frame.copy()
            cv2.fillPoly(overlay, fill_poly, color=(255, 165, 0))  # Turuncu
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Şerit çizgilerini çiz
            if lp1 and lp2:
                cv2.line(frame, lp1, lp2, (0, 255, 255), 3)  # Sol şerit
            if rp1 and rp2:
                cv2.line(frame, rp1, rp2, (0, 255, 255), 3)  # Sağ şerit
    else:
        pass
    
    # ROI noktalarını daha modern bir şekilde göster
    for pt in vertices[0]:
        # Dış halka
        cv2.circle(
            frame,
            center=(int(pt[0]), int(pt[1])),
            radius=8,
            color=(255, 255, 255),    # Beyaz dış halka
            thickness=2
        )
        # İç daire
        cv2.circle(
            frame,
            center=(int(pt[0]), int(pt[1])),
            radius=4,
            color=(0, 165, 255),    # Turuncu iç daire
            thickness=-1
        )
    
    # Sonucu birleştir ve döndür
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    # Görüntüyü biraz daha parlak yap
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=10)
    
    return result

# Video Kaynağı (Dosya yolu veya 0 for webcam)
video_path = "deneme_video/Araba.mp4" 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    cv2.imshow("Modern Lane Detection", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
