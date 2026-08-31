import cv2
import numpy as np

def find_largest_valid_contour(mask, min_area=100):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    valid = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    if not valid:
        return None

    return max(
        valid,
        key=cv2.contourArea
    )


def grabcut_segmentation(crop):

    h, w = crop.shape[:2]

    if h < 20 or w < 20:
        return None

    mask = np.zeros(
        (h, w),
        np.uint8
    )

    # Border = probable background
    mask[:, :] = cv2.GC_BGD

    margin_x = max(
        2,
        int(w * 0.05)
    )

    margin_y = max(
        2,
        int(h * 0.05)
    )

    # Interior = probable foreground
    mask[
        margin_y:h-margin_y,
        margin_x:w-margin_x
    ] = cv2.GC_PR_FGD

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            crop,
            mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        return None

    binary = np.where(
        (
            (mask == cv2.GC_FGD)
            |
            (mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contour = find_largest_valid_contour(
        binary
    )

    if contour is None:
        return None

    return binary, contour


def hsv_segmentation(crop):

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    # General foreground mask.
    # This is deliberately broad for the prototype.
    mask = cv2.inRange(
        hsv,
        np.array([0, 20, 20]),
        np.array([179, 255, 250])
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contour = find_largest_valid_contour(
        mask
    )

    if contour is None:
        return None

    return mask, contour


def analyze_onion(crop, debug=False):

    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]

    if h < 20 or w < 20:
        return None

    # ==========================================
    # Try GrabCut first
    # ==========================================

    result = grabcut_segmentation(crop)

    method = "GrabCut"

    # ==========================================
    # Fallback to HSV
    # ==========================================

    if result is None:

        result = hsv_segmentation(crop)

        method = "HSV"

    if result is None:

        return None

    mask, contour = result

    # ==========================================
    # Area
    # ==========================================

    area = cv2.contourArea(
        contour
    )

    # ==========================================
    # Perimeter
    # ==========================================

    perimeter = cv2.arcLength(
        contour,
        True
    )

    # ==========================================
    # Circularity
    # ==========================================

    if perimeter > 0:

        circularity = (
            4
            * np.pi
            * area
            / (perimeter ** 2)
        )

    else:

        circularity = 0

    # ==========================================
    # Bounding box
    # ==========================================

    x, y, width, height = cv2.boundingRect(
        contour
    )

    # ==========================================
    # Aspect ratio
    # ==========================================

    if height > 0:

        aspect_ratio = (
            width / height
        )

    else:

        aspect_ratio = 0

    # ==========================================
    # Color
    # ==========================================

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    mean_hsv = cv2.mean(
        hsv,
        mask=mask
    )

    mean_hue = mean_hsv[0]

    mean_saturation = mean_hsv[1]

    mean_value = mean_hsv[2]

    # ==========================================
    # Debug image
    # ==========================================

    debug_image = crop.copy()

    cv2.drawContours(
        debug_image,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        debug_image,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )

    if debug:

        cv2.imshow(
            f"Mask - {method}",
            mask
        )

        cv2.imshow(
            f"Contour - {method}",
            debug_image
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    # ==========================================
    # Return
    # ==========================================

    return {

        "method": method,

        "area": round(
            float(area),
            2
        ),

        "width": int(width),

        "height": int(height),

        "aspect_ratio": round(
            float(aspect_ratio),
            3
        ),

        "circularity": round(
            float(circularity),
            3
        ),

        "mean_hue": round(
            float(mean_hue),
            2
        ),

        "mean_saturation": round(
            float(mean_saturation),
            2
        ),

        "mean_brightness": round(
            float(mean_value),
            2
        )
    }
