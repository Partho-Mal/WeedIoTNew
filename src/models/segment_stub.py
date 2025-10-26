# src/models/segment_stub.py
import cv2
import numpy as np
from src.preprocessing import (
    compute_ndvi_from_bgr, 
    threshold_mask_from_ndvi,
    detect_weeds_by_size_color,
    detect_weeds_texture_based
)

def segment(img_bgr, method='ndvi', threshold=0.12):
    """
    Segment WEEDS specifically from image.
    
    Methods:
    - 'ndvi': Simple vegetation detection (detects ALL plants)
    - 'color': HSV-based weed detection (size + color filtering)
    - 'texture': Texture-based weed detection (irregular patterns)
    - 'size_filter': Color + size-based weed detection
    
    Args:
        img_bgr: Input image in BGR format (OpenCV default)
        method: Segmentation method
        threshold: Threshold value for segmentation
        
    Returns:
        mask: Binary mask (255 = WEED, 0 = crop/soil)
        aux: Dictionary with auxiliary information
    """
    if method == 'ndvi':
        # Simple NDVI - detects ALL vegetation
        # Note: This doesn't distinguish crops from weeds!
        ndvi = compute_ndvi_from_bgr(img_bgr)
        mask = threshold_mask_from_ndvi(ndvi, thresh=threshold)
        aux = {'ndvi': ndvi, 'note': 'Detects all vegetation, not just weeds'}
        
    elif method == 'color':
        # Weed-specific detection using size and color
        # Assumes crops are larger and more uniformly green
        min_area = int(50 * (1.0 - threshold))  # Adjust with threshold
        max_area = 5000
        mask = detect_weeds_by_size_color(img_bgr, min_area=min_area, max_area=max_area)
        aux = {'method': 'size_color', 'min_area': min_area, 'max_area': max_area}
        
    elif method == 'texture':
        # Texture-based weed detection
        mask = detect_weeds_texture_based(img_bgr)
        aux = {'method': 'texture_based'}

    elif method == 'size_filter':
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 30, 30])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        min_area = 30
        max_area = 3000
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        # 🔥 normalize here so downstream heatmap works
        mask = filtered_mask.astype(np.float32) / 255.0
        
        aux = {'method': 'size_filtered', 'note': 'normalized 0-1 for heatmap'}

        
    # elif method == 'size_filter':
    #     # Enhanced color-based with morphological filtering
    #     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
    #     # Broader color range to catch various weed types
    #     lower = np.array([25, 30, 30])   # Yellow-green to brown weeds
    #     upper = np.array([85, 255, 255])
    #     mask = cv2.inRange(hsv, lower, upper)
        
    #     # Morphological operations to clean up
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
    #     # Remove very large blobs (likely crops) and very small noise
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     filtered_mask = np.zeros_like(mask)
        
    #     min_area = 30
    #     max_area = 3000
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if min_area < area < max_area:
    #             cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
    #     mask = filtered_mask
    #     aux = {'hsv': hsv, 'method': 'size_filtered'}
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ndvi', 'color', 'texture', or 'size_filter'")
    
    return mask, aux


def segment_with_crop_rows(img_bgr, row_spacing=50, row_tolerance=15):
    """
    Advanced weed detection for row crops.
    Detects vegetation between crop rows as weeds.
    
    Args:
        img_bgr: Input image
        row_spacing: Expected pixel spacing between crop rows
        row_tolerance: Pixel tolerance for crop row width
        
    Returns:
        weed_mask: Binary mask (255 = weed)
        aux: Detection metadata
    """
    # Get all vegetation
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255
    
    # Analyze horizontal projection to find crop rows
    h, w = veg_mask.shape
    vertical_projection = np.sum(veg_mask > 0, axis=0) / h
    
    # Smooth to find peaks
    smoothed = np.convolve(vertical_projection, np.ones(10)/10, mode='same')
    
    # Simple peak detection
    threshold_val = np.mean(smoothed) + 0.5 * np.std(smoothed)
    peaks = []
    in_peak = False
    start = 0
    
    for i, val in enumerate(smoothed):
        if val > threshold_val and not in_peak:
            in_peak = True
            start = i
        elif val <= threshold_val and in_peak:
            in_peak = False
            peaks.append((start + i) // 2)  # Middle of peak
    
    # Create crop row mask
    crop_row_mask = np.zeros_like(veg_mask)
    for peak in peaks:
        x_start = max(0, peak - row_tolerance)
        x_end = min(w, peak + row_tolerance)
        crop_row_mask[:, x_start:x_end] = 255
    
    # Weeds = vegetation NOT in crop rows
    weed_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(crop_row_mask))
    
    aux = {
        'crop_rows': peaks,
        'num_rows': len(peaks),
        'vegetation_mask': veg_mask,
        'crop_row_mask': crop_row_mask
    }
    
    return weed_mask, aux