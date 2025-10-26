# src/preprocessing.py
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def compute_ndvi_from_bgr(img_bgr):
    """
    Compute a vegetation index from BGR image.
    Returns vegetation index in range [-1, 1]. 
    """
    img = img_bgr.astype(float)
    b, g, r = cv2.split(img)
    
    nir_proxy = g  # Green channel (vegetation)
    red = r        # Red channel (soil)
    
    denom = (nir_proxy + red + 1e-6)
    ndvi = (nir_proxy - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi

def threshold_mask_from_ndvi(ndvi, thresh=0.05):
    """
    Return binary mask (255 = vegetation, 0 = background).
    This detects ALL vegetation (crops + weeds).
    """
    mask = (ndvi > thresh).astype('uint8') * 255
    return mask

def detect_weeds_by_row_crops(img_bgr, row_spacing=50, row_tolerance=15):
    """
    Detect weeds in row-crop agriculture.
    Assumes crops are planted in regular rows.
    Anything between rows is considered a weed.
    
    Args:
        img_bgr: Input BGR image
        row_spacing: Expected spacing between crop rows (pixels)
        row_tolerance: Tolerance for crop row width (pixels)
    
    Returns:
        weed_mask: Binary mask (255 = weed, 0 = crop/soil)
    """
    # Get all vegetation
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255
    
    # Find crop rows by analyzing vegetation density along horizontal lines
    h, w = veg_mask.shape
    density = np.sum(veg_mask > 0, axis=0) / h  # Column-wise density
    
    # Smooth density to find peaks (crop rows)
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(density, sigma=10)
    
    # Find peaks (crop rows)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smoothed, distance=row_spacing//2, prominence=0.1)
    
    # Create crop row mask
    crop_row_mask = np.zeros_like(veg_mask)
    for peak in peaks:
        x_start = max(0, peak - row_tolerance)
        x_end = min(w, peak + row_tolerance)
        crop_row_mask[:, x_start:x_end] = 255
    
    # Weeds = vegetation outside crop rows
    weed_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(crop_row_mask))
    
    return weed_mask

def detect_weeds_by_size_color(img_bgr, min_area=50, max_area=5000):
    """
    Detect weeds using size and color differences from crops.
    Assumes crops are larger, more uniform green than weeds.
    
    Args:
        img_bgr: Input BGR image
        min_area: Minimum area for weed detection (pixels)
        max_area: Maximum area for weed detection (pixels)
    
    Returns:
        weed_mask: Binary mask (255 = weed, 0 = crop/soil)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Detect healthy crop (strong green, high saturation)
    crop_lower = np.array([40, 80, 60])   # Bright green
    crop_upper = np.array([75, 255, 255])
    crop_mask = cv2.inRange(hsv, crop_lower, crop_upper)
    
    # Detect weeds (broader range, includes yellowish/brownish weeds)
    weed_lower = np.array([20, 30, 30])   # Yellow-green to brown
    weed_upper = np.array([90, 255, 255])
    potential_weed = cv2.inRange(hsv, weed_lower, weed_upper)
    
    # Remove crop areas from potential weeds
    weed_mask = cv2.bitwise_and(potential_weed, cv2.bitwise_not(crop_mask))
    
    # Filter by size: weeds are typically smaller than crops
    contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(weed_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
    return filtered_mask

def detect_weeds_texture_based(img_bgr):
    """
    Detect weeds using texture analysis.
    Crops have uniform texture, weeds have irregular patterns.
    
    Returns:
        weed_mask: Binary mask (255 = weed, 0 = crop/soil)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Get vegetation areas
    ndvi = compute_ndvi_from_bgr(img_bgr)
    veg_mask = (ndvi > 0.05).astype('uint8') * 255
    
    # Calculate local variance (texture measure)
    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = sqr_mean - mean**2
    
    # High variance = irregular texture = potential weed
    # Low variance = uniform = crop
    texture_thresh = np.percentile(variance[veg_mask > 0], 60)
    high_variance_mask = (variance > texture_thresh).astype('uint8') * 255
    
    # Combine with vegetation mask
    weed_mask = cv2.bitwise_and(veg_mask, high_variance_mask)
    
    return weed_mask

def compute_color_based_mask(img_bgr, lower_green, upper_green):
    """
    Direct color-based segmentation in HSV space.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def resize_keep_aspect(img, max_side=512):
    """Resize image keeping aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return new

def combined_weed_heatmap(img):
    """
    Compute a single heatmap combining multiple weed detection methods
    """
    # Import functions from the same module
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=float)

    # 1️⃣ Color-based
    color_mask = detect_weeds_by_size_color(img)
    heat += color_mask / 255.0  # normalize 0-1

    # 2️⃣ Row-crop based
    row_mask = detect_weeds_by_row_crops(img)
    heat += row_mask / 255.0

    # 3️⃣ Texture-based
    texture_mask = detect_weeds_texture_based(img)
    heat += texture_mask / 255.0

    # Optional: normalize final heatmap to 0-1
    heat = np.clip(heat / 3.0, 0, 1)
    return heat
# -------------------------