"""
Test script to compare crop detection vs weed detection methods.
Shows why NDVI alone isn't enough for weed-specific detection.

Usage: python test_weed_detection.py path/to/field/image.jpg
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def compute_ndvi_from_bgr(img_bgr):
    """Fixed NDVI using green-red difference"""
    img = img_bgr.astype(float)
    b, g, r = cv2.split(img)
    nir_proxy = g
    red = r
    denom = (nir_proxy + red + 1e-6)
    ndvi = (nir_proxy - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

def detect_all_vegetation(img_bgr):
    """Detects ALL vegetation (crops + weeds)"""
    ndvi = compute_ndvi_from_bgr(img_bgr)
    return (ndvi > 0.05).astype('uint8') * 255

def detect_weeds_only(img_bgr):
    """Detect ONLY weeds using size and color filtering"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Detect ALL vegetation first
    veg_lower = np.array([25, 30, 30])
    veg_upper = np.array([85, 255, 255])
    all_veg = cv2.inRange(hsv, veg_lower, veg_upper)
    
    # Detect healthy crops (bright, uniform green)
    crop_lower = np.array([40, 80, 60])
    crop_upper = np.array([75, 255, 255])
    crops = cv2.inRange(hsv, crop_lower, crop_upper)
    
    # Weeds = all vegetation - crops
    weeds_only = cv2.bitwise_and(all_veg, cv2.bitwise_not(crops))
    
    # Size filtering: remove large blobs (crops) and tiny noise
    contours, _ = cv2.findContours(weeds_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(weeds_only)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 3000:  # Weed size range
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    
    # Clean up with morphology
    kernel = np.ones((3, 3), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    
    return filtered

def create_synthetic_field():
    """Create a synthetic field with crops in rows and scattered weeds"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Brown soil background
    img[:, :] = [80, 110, 90]  # BGR: brownish soil
    
    # Add crop rows (large, uniform green plants)
    crop_positions = [100, 200, 300, 400, 500]  # 5 crop rows
    for x in crop_positions:
        for y in range(50, 350, 40):  # Regular spacing
            # Large healthy crop plants
            cv2.circle(img, (x, y), 15, (50, 180, 70), -1)  # BGR: bright green
            cv2.circle(img, (x, y), 12, (60, 200, 80), -1)  # Brighter center
    
    # Add weeds (smaller, scattered, varied color)
    np.random.seed(42)
    for _ in range(30):
        x = np.random.randint(20, 580)
        y = np.random.randint(20, 380)
        
        # Avoid crop row centers
        if min([abs(x - cx) for cx in crop_positions]) > 25:
            # Smaller, yellowish-green weeds
            size = np.random.randint(5, 12)
            color = [
                np.random.randint(40, 80),   # B
                np.random.randint(120, 160), # G (less green than crops)
                np.random.randint(60, 100)   # R (more yellow/brown)
            ]
            cv2.circle(img, (x, y), size, color, -1)
    
    return img

def test_weed_detection(image_path):
    """Compare vegetation detection vs weed-specific detection"""
    # Read or create image
    if image_path == 'synthetic':
        img = create_synthetic_field()
        cv2.imwrite('synthetic_field_with_weeds.jpg', img)
        print("✓ Created synthetic field image: synthetic_field_with_weeds.jpg")
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read {image_path}")
            return
    
    print(f"\nImage shape: {img.shape}")
    
    # Method 1: Detect all vegetation (NDVI-like)
    print("\n=== Method 1: NDVI (All Vegetation) ===")
    all_veg_mask = detect_all_vegetation(img)
    veg_pixels = np.sum(all_veg_mask > 0)
    veg_percent = 100 * veg_pixels / (img.shape[0] * img.shape[1])
    print(f"Vegetation detected: {veg_percent:.2f}% of field")
    print("⚠️  This includes BOTH crops AND weeds!")
    
    # Method 2: Detect only weeds
    print("\n=== Method 2: Weed-Specific Detection ===")
    weed_mask = detect_weeds_only(img)
    weed_pixels = np.sum(weed_mask > 0)
    weed_percent = 100 * weed_pixels / (img.shape[0] * img.shape[1])
    print(f"Weeds detected: {weed_percent:.2f}% of field")
    print("✓ This targets only weeds (smaller, scattered plants)")
    
    # Calculate savings
    print("\n=== Herbicide Savings Analysis ===")
    if veg_pixels > 0:
        savings = 100 * (1 - weed_pixels / veg_pixels)
        print(f"Herbicide savings: {savings:.1f}%")
        print(f"  • Spraying all vegetation: {veg_percent:.2f}% of field")
        print(f"  • Spraying only weeds: {weed_percent:.2f}% of field")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and vegetation masks
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Field Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(all_veg_mask, cmap='Greens')
    axes[0, 1].set_title(f'NDVI: All Vegetation ({veg_percent:.1f}%)\n⚠️ Crops + Weeds')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(weed_mask, cmap='Reds')
    axes[0, 2].set_title(f'Weed-Specific Detection ({weed_percent:.1f}%)\n✓ Only Weeds')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    overlay_all = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay_all[all_veg_mask > 0] = [0, 255, 0]  # Green
    axes[1, 0].imshow(cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, overlay_all, 0.4, 0))
    axes[1, 0].set_title('NDVI Overlay (Green = All Vegetation)')
    axes[1, 0].axis('off')
    
    overlay_weed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay_weed[weed_mask > 0] = [255, 0, 0]  # Red
    axes[1, 1].imshow(cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, overlay_weed, 0.4, 0))
    axes[1, 1].set_title('Weed Overlay (Red = Weeds Only)')
    axes[1, 1].axis('off')
    
    # Comparison
    comparison = np.zeros_like(img)
    comparison[all_veg_mask > 0] = [0, 255, 0]  # Green = all vegetation
    comparison[weed_mask > 0] = [255, 0, 0]      # Red = weeds (overwrites)
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title('Comparison\nGreen = Crops, Red = Weeds')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('weed_detection_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results saved to 'weed_detection_comparison.png'")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No image provided. Creating synthetic field for demonstration...\n")
        test_weed_detection('synthetic')
    else:
        test_weed_detection(sys.argv[1])