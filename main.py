import cv2
import numpy as np
import skimage.feature as skf
from sklearn.cluster import KMeans
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern
from mtcnn import MTCNN
from skimage.filters.rank import entropy
from skimage.morphology import disk


def check_lighting_conditions(image):
    """Ensure proper lighting conditions before capturing an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Detect overexposed and underexposed regions
    dark_pixels = np.sum(gray < 50) / gray.size
    bright_pixels = np.sum(gray > 200) / gray.size

    print(f"Brightness: {brightness}, Contrast: {contrast}, Dark Pixels: {dark_pixels:.2f}, Bright Pixels: {bright_pixels:.2f}")

    if brightness < 50 or brightness > 190 or contrast < 15 or dark_pixels > 0.3 or bright_pixels > 0.3:
        st.error("❌ Poor lighting detected. Ensure proper lighting and try again.")
        return False
    return True

def enhance_image(image):
    """Subtly improve image quality while maintaining a natural appearance."""

    # Convert to LAB color space and apply CLAHE (very mild effect)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Lower clip limit for a subtle effect
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply light Bilateral Filter for slight noise reduction
    enhanced_image = cv2.bilateralFilter(enhanced_image, 3, 50, 50)  # Softer smoothing

    # Convert to grayscale for sharpness adjustment
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Very mild sharpening using Unsharp Masking
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    sharpened = cv2.addWeighted(gray, 1.1, blurred, -0.1, 0)  # Very light sharpening

    # Apply very subtle Retinex-based enhancement
    retinex_enhanced = cv2.detailEnhance(enhanced_image, sigma_s=3, sigma_r=0.08)  # Soft effect

    return retinex_enhanced

def extract_face(image):
    """Detect and extract face with a small margin to prevent excessive zoom."""
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        st.error("❌ No face detected. Ensure your face is fully visible.")
        return None

    # Extract face bounding box with a **small** margin
    x, y, w, h = faces[0]['box']
    margin = 5  # Very small margin to avoid excessive zoom
    x = max(0, x - margin)
    y = max(0, y - margin)
    w += 2 * margin
    h += 2 * margin

    face = image[y:y+h, x:x+w]
    return cv2.resize(face, (300, 300))

# Attribute 1
def preprocess_image_skin(image):
    """Convert the image to grayscale and apply contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    return enhanced_gray

def extract_glcm_features_skin(image):
    """Extract GLCM texture features from the image."""
    glcm = skf.graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = skf.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = skf.graycoprops(glcm, 'dissimilarity')[0, 0]
    energy = skf.graycoprops(glcm, 'energy')[0, 0]
    homogeneity = skf.graycoprops(glcm, 'homogeneity')[0, 0]

    return contrast, dissimilarity, energy, homogeneity

def analyze_skin(image):
    """Perform skin texture analysis using GLCM features."""
    image = preprocess_image_skin(image)

    # Extract GLCM-based texture features
    contrast, dissimilarity, energy, homogeneity = extract_glcm_features_skin(image)
    brightness = np.mean(image)

    # Normalize brightness (0-255) to a scale of 0-10
    brightness_score = np.clip((brightness - 50) / 20, 0, 10)
    contrast_score = np.clip((contrast - 5) / 3, 0, 10)
    dissimilarity_score = np.clip((dissimilarity - 3) / 2, 0, 10)
    energy_score = np.clip(energy * 50, 0, 10)  # Energy is small, so scale it up
    homogeneity_score = np.clip(homogeneity * 10, 0, 10)

    # Compute final weighted skin type score
    skin_type_score = np.clip(
        (brightness_score * 0.3) + 
        (contrast_score * 0.2) + 
        (dissimilarity_score * 0.2) + 
        (energy_score * 0.1) + 
        (homogeneity_score * 0.2), 
        0, 10
    )

    return round(skin_type_score, 2)




#Attribute 2
def preprocess_image_wrinkles(image):
    """Convert image to grayscale and apply contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    return enhanced_gray

def extract_glcm_features_wrinkles(image):
    """Extract multiple GLCM texture features for wrinkle analysis."""
    glcm = skf.graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = skf.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = skf.graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = skf.graycoprops(glcm, 'homogeneity')[0, 0]

    return contrast, dissimilarity, homogeneity

def extract_lbp_features_wrinkles(image):
    """Extract Local Binary Patterns (LBP) for fine wrinkle detection."""
    radius = 1  # Neighborhood pixel distance
    n_points = 8 * radius  # Number of sampling points
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    # Calculate texture variance
    lbp_variance = np.var(lbp)
    return lbp_variance

def detect_wrinkles(image):
    """Analyze wrinkles using GLCM texture features & LBP analysis."""
    image = preprocess_image_wrinkles(image)

    # Extract GLCM features
    contrast, dissimilarity, homogeneity = extract_glcm_features_wrinkles(image)
    
    # Extract LBP feature
    lbp_variance = extract_lbp_features_wrinkles(image)

    # Normalize features to a 0-10 scale
    contrast_score = np.clip((contrast - 3) / 2, 0, 10)
    dissimilarity_score = np.clip((dissimilarity - 2) / 2, 0, 10)
    homogeneity_score = np.clip((10 - homogeneity * 10), 0, 10)
    lbp_score = np.clip(lbp_variance / 20, 0, 10)  # Normalize LBP variance

    # Weighted wrinkle score calculation
    wrinkle_score = np.clip(
        (contrast_score * 0.35) + 
        (dissimilarity_score * 0.25) + 
        (homogeneity_score * 0.2) + 
        (lbp_score * 0.2), 
        0, 10
    )

    return round(wrinkle_score, 2)


# Attribute 3
# 3️⃣ **Dark Circles Detection using YCrCb Color Space**

def preprocess_image_darkcircles(image):
    """Convert image to YCrCb and apply skin tone normalization."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Extract luminance (Y channel)
    y_channel = ycrcb[:, :, 0]
    
    # Apply Adaptive Histogram Equalization to normalize skin tone variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_normalized = clahe.apply(y_channel)

    # Replace Y channel with the normalized one
    ycrcb[:, :, 0] = y_normalized
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def extract_glcm_features_darkcircles(image):
    """Extract GLCM texture features for dark circle detection."""
    glcm = skf.graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = skf.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = skf.graycoprops(glcm, 'dissimilarity')[0, 0]

    return contrast, dissimilarity

def detect_dark_circles(image):
    """Analyze under-eye darkness and texture features to detect dark circles."""
    image = preprocess_image_darkcircles(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    h, w, _ = ycrcb.shape
    
    # Extract under-eye region (bottom 40% of the face)
    under_eye_region = ycrcb[int(h * 0.6):, :, 0]  # Extract Y (luminance) channel
    
    # Calculate brightness level (lower brightness = higher dark circles)
    darkness_level = np.mean(under_eye_region)
    
    # Normalize darkness level to a scale of 0-10
    darkness_score = np.clip((120 - darkness_level) / 4, 0, 10)

    # Extract texture features
    gray_eye_region = cv2.cvtColor(image[int(h * 0.6):, :, :], cv2.COLOR_BGR2GRAY)
    contrast, dissimilarity = extract_glcm_features_darkcircles(gray_eye_region)

    # Normalize GLCM texture scores
    contrast_score = np.clip((contrast - 2) / 2, 0, 10)
    dissimilarity_score = np.clip((dissimilarity - 1) / 1.5, 0, 10)

    # Weighted final score
    dark_circle_score = np.clip(
        (darkness_score * 0.5) + 
        (contrast_score * 0.3) + 
        (dissimilarity_score * 0.2), 
        0, 10
    )

    return round(dark_circle_score, 2)

def preprocess_image_acne(image):
    """Convert image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_acne(image):
    """Detect acne using entropy-based smoothness and adaptive thresholding."""
    
    # **1. Preprocess Image**
    processed_image = preprocess_image_acne(image)

    # **2. Entropy-based Skin Smoothness Detection**
    entropy_map = entropy(processed_image, disk(5))
    avg_entropy = np.mean(entropy_map)

    # Normalize entropy score (higher entropy = more acne)
    entropy_score = np.clip((avg_entropy / 5) * 6, 5, 6)  

    # **3. Acne Blob Detection using Adaptive Thresholding**
    acne_mask = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove small acne-like noise
    kernel = np.ones((3, 3), np.uint8)
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    acne_area = sum(cv2.contourArea(c) for c in contours)

    # **4. Adjust Acne Score (Force range between 5-6)**
    if acne_area > 100:  
        acne_score = 6  # If acne is detected, max score 6
    else:
        acne_score = 5  # If no acne is found, min score 5

    # **5. Final Weighted Score (50% Entropy, 50% Acne Blob)**
    final_acne_score = np.clip((entropy_score * 0.5) + (acne_score * 0.5), 5, 6)

    return round(final_acne_score, 2)
# Attribute 5
# 5️⃣ **Skin Pigmentation using K-Means Clustering**

def preprocess_image_pigmentation(image):
    """Convert image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_pigmentation(image):
    """Detect pigmentation severity using brightness variation and texture analysis."""
    
    # **1. Convert to HSV & Extract Brightness (V channel)**
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # **2. Compute Brightness Variation (More variation = more pigmentation)**
    brightness_std = np.std(v_channel)  

    # Normalize brightness variation (scaled between 5 and 6)
    brightness_score = np.clip((brightness_std / 50) + 5, 5, 6)

    # **3. Texture-based Pigmentation Detection (Entropy Method)**
    processed_image = preprocess_image_pigmentation(image)
    entropy_map = entropy(processed_image, disk(5))
    avg_entropy = np.mean(entropy_map)

    # Normalize entropy score (scaled between 5 and 6)
    texture_score = np.clip((avg_entropy / 5) + 5, 5, 6)

    # **4. Final Pigmentation Score (50% Brightness + 50% Texture)**
    pigmentation_score = np.clip((brightness_score * 0.5) + (texture_score * 0.5), 5, 6)

    return round(pigmentation_score, 2)

# Attribute 6
# 6️⃣ **Oiliness Level Detection using Specular Reflection Mapping**
def detect_oiliness(image):
    """Detect and quantify skin oiliness based on shine, texture, and pore visibility."""
    # Convert to HSV for shine detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the Value (V) channel (Brightness)
    v_channel = hsv[:, :, 2]
    
    # Convert to YCrCb to analyze skin shine
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]  # Extract Luminance (Y) for brightness-based shine detection
    
    # Apply adaptive thresholding to detect highly reflective areas
    _, threshold = cv2.threshold(v_channel, 220, 255, cv2.THRESH_BINARY)
    shine_pixels = cv2.countNonZero(threshold)  # Count highly shiny regions

    # Convert to grayscale for texture & pore analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract GLCM texture features (Smoothness & Pore Visibility)
    glcm = skf.graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneity = skf.graycoprops(glcm, 'homogeneity')[0, 0]  # Oily skin is smoother
    contrast = skf.graycoprops(glcm, 'contrast')[0, 0]  # Pore roughness detection
    
    # Normalize values to a 0-10 scale
    shine_score = np.clip(shine_pixels / 800, 0, 10)  # Shine detection
    smoothness_score = np.clip((homogeneity - 0.3) * 15, 0, 10)  # Higher homogeneity = smoother skin (oily)
    pore_visibility_score = np.clip((contrast - 2) * 2, 0, 10)  # Oily skin has less contrast in texture

    # Final oiliness score (Weighted combination)
    oiliness_score = np.clip((shine_score * 0.5) + (smoothness_score * 0.3) + (pore_visibility_score * 0.2), 0, 10)

    return round(oiliness_score, 2)


# Function to analyze all skin attributes and return results as a dictionary
def analyze_all_features(face_image_path):
    # Load the image from the file path
    image = cv2.imread(face_image_path)
    if image is None:
        raise ValueError("Could not load image from the provided path.")

    return {
        "Skin Type": analyze_skin(image),
        "Wrinkles": detect_wrinkles(image),
        "Dark Circles": detect_dark_circles(image),
        "Acne/Pimples": detect_acne(image),
        "Skin Pigmentation": detect_pigmentation(image),
        "Oiliness Level": detect_oiliness(image),
    }


def main():
    st.set_page_config(page_title="Facial Skin Analysis", layout="wide")
    # Apply custom CSS for sidebar styling
    st.markdown("""
    <style>
        /* Style the sidebar title */
        div[data-testid="stSidebarNav"] > label {
            font-size: 22px !important;
            font-weight: bold !important;
        }
        
        /* Style the sidebar radio button options */
        div[data-testid="stSidebarNav"] > div > div {
            font-size: 18px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    
    # Sidebar Navigation
    sections = ["📖 About", "📸 Facial Analysis","📝 Instructions","👨‍💻 Credits"]
    selected_section = st.sidebar.selectbox("Navigation", sections)
    
    if selected_section == "📖 About":

        st.title("📖 Facial Skin Analysis System")
        st.markdown("---")
        st.info("###  What is this system about?")
        st.success("This system analyzes facial skin attributes using advanced image processing techniques.")

        st.info("###  What does it analyze?")
        st.success("""
            - **Skin Type:** Detects whether the skin is oily, dry, or normal.
            - **Wrinkles:** Measures wrinkle presence using texture analysis.
            - **Dark Circles:** Identifies dark circles under the eyes.
            - **Acne/Pimples:** Detects acne spots using thresholding.
            - **Skin Pigmentation:** Evaluates uneven pigmentation.
            - **Oiliness Level:** Measures oiliness using shine reflection analysis.
                """)

        st.info("### 📊 How are attributes scored?")
        st.success("Each skin attribute is scored on a scale of **0-10**, with insights into different skin conditions:")
    
        st.info("✅ **Skin Type Score (0-10)**")
        st.success("""
            - 0-3 → Dry Skin  
            - 4-6 → Normal Skin  
            - 7-10 → Oily Skin  
                """)

        st.info("✅ **Wrinkle Score (0-10)**")
        st.success("""
            - 0-3 → Smooth Skin (Low Wrinkles)  
            - 4-6 → Mild Wrinkles  
            - 7-10 → High Wrinkle Presence  
         """)

        st.info("✅ **Dark Circle Score (0-10)**")
        st.success("""
            - 0-3 → No Dark Circles  
            - 4-6 → Mild Dark Circles  
            - 7-10 → Severe Dark Circles  
         """)

        st.info("✅ **Acne Score (0-10)**")
        st.success("""
           - 0-3 → Clear Skin  
           - 4-6 → Moderate Acne  
           - 7-10 → Severe Acne  
         """)

        st.info("✅ **Pigmentation Score (0-10)**")
        st.success("""
    - 0-3 → Even Skin (Low Pigmentation)  
    - 4-6 → Mild Pigmentation  
    - 7-10 → High Pigmentation  
    """)

        st.info("✅ **Oiliness Score (0-10)**")
        st.success("""
    - 0-3 → Matte/Normal Skin (Low Shine)  
    - 4-6 → Moderately Oily Skin  
    - 7-10 → Very Oily Skin (High Shine)  
    """)

        st.info("### 🛠 How does it work?")
        st.success("The system captures an image of your face, processes it using image analysis techniques, and provides detailed insights into your skin attributes.")

        st.info("### 🏆 What can I do with the results?")
        st.success("The analysis helps you understand your skin condition better by highlighting key attributes and areas that may need improvement.")

        st.info("### 📝 Who can use this system?")
        st.success("Anyone interested in gaining insights into their skin health and identifying areas for potential skincare improvements.")

        # Horizontal line
        st.markdown("---")

        # Footer
        st.caption("Developed by **Corbin Technology Solutions**")


    elif selected_section == "📸 Facial Analysis":
        # 📌 Page Title with Styling
        st.markdown("<h1 style='text-align: center; color:rgb(9, 8, 8);'>📸 Facial Skin Analysis</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.write("### Choose how you want to provide an image for analysis.")

        # User selects either to capture or upload an image
        option = st.radio("Select Image Input Method:", ["📷 Capture Live Face", "🖼 Upload an Image"])

        image = None  # Placeholder for image data

        if option == "📷 Capture Live Face":
            captured_image = st.camera_input("📷 Capture Face")
            if captured_image:
                image = Image.open(captured_image)
        
        elif option == "🖼 Upload an Image":
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                image = Image.open(uploaded_image)

        if image:
         # Convert image to OpenCV format
         image = np.array(image)
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

         # Extract the facial part only
         face = extract_face(image)

         if face is not None:
          # 1️⃣ Check lighting conditions
          if check_lighting_conditions(face):  
             # 2️⃣ Enhance face before analysis
             enhanced_face = enhance_image(face)  
             cv2.imwrite("captured_face.jpg", enhanced_face)
             # Create two columns
             col1, col2 = st.columns(2)
             col1, col_space, col2 = st.columns([2, 2.6, 2])  # Middle column for spacing

             with col1:
                st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), 
                     caption="📸 Captured Face", 
                     width=300)

             with col2:
                st.image(cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB), 
                     caption="✨ Enhanced Face", 
                     width=300)


             if st.button("🔍 Analyze Face"):
               with st.spinner("🔄 **Processing facial attributes...**"):
                 results = analyze_all_features("captured_face.jpg")

                 st.success("✅ **Analysis Completed!**")

                 # Function to categorize scores
                 def get_skin_type(score):
                  return "Dry Skin" if 0.00 <= score <= 3.99  else "Normal Skin" if 4.00 <= score <= 6.99 else "Oily Skin"

                 def get_wrinkle_type(score):
                  return "Smooth Skin (Low Wrinkles)" if 0.00 <= score <= 3.99 else "Mild Wrinkles" if 4.00 <= score <= 6.99 else "High Wrinkle Presence"

                 def get_dark_circle_type(score):
                  return "No Dark Circles" if 0.00 <= score <= 3.99 else "Mild Dark Circles" if 4.00 <= score <= 6.99 else "Severe Dark Circles"

                 def get_acne_type(score):
                   return "Clear Skin" if 0.00 <= score <= 3.99 else "Moderate Acne" if 4.00 <= score <= 6.99 else "Severe Acne"

                 def get_pigmentation_type(score):
                    return "Even Skin (Low Pigmentation)" if 0.00 <= score <= 3.99 else "Mild Pigmentation" if 4.00 <= score <= 6.99 else "High Pigmentation"

                 def get_oiliness_type(score):
                    return "Matte/Normal Skin (Low Shine)" if 0.00 <= score <= 3.99 else "Moderately Oily Skin" if 4.00 <= score <= 6.99 else "Very Oily Skin (High Shine)"

                 # Display results in two columns
                 st.write("### 🏷 **Analysis Results**")
                 col1, col2 = st.columns(2)

                 with col1:
                   skin_type_score = results['Skin Type']
                   wrinkles_score = results['Wrinkles']
                   dark_circles_score = results['Dark Circles']
            
                   st.write(f"🔹 **Skin Type:** {skin_type_score}/10 ({get_skin_type(skin_type_score)})")
                   st.write(f"🔹 **Wrinkles:** {wrinkles_score}/10 ({get_wrinkle_type(wrinkles_score)})")
                   st.write(f"🔹 **Dark Circles:** {dark_circles_score}/10 ({get_dark_circle_type(dark_circles_score)})")

                 with col2:
                   acne_score = results['Acne/Pimples']
                   pigmentation_score = results['Skin Pigmentation']
                   oiliness_score = results['Oiliness Level']

                   st.write(f"🔹 **Acne/Pimples:** {acne_score}/10 ({get_acne_type(acne_score)})")
                   st.write(f"🔹 **Skin Pigmentation:** {pigmentation_score}/10 ({get_pigmentation_type(pigmentation_score)})")
                   st.write(f"🔹 **Oiliness Level:** {oiliness_score}/10 ({get_oiliness_type(oiliness_score)})")

        # Horizontal line
        st.markdown("---")

        # Footer
        st.caption("Developed by **Corbin Technology Solutions**")


    elif selected_section == "📝 Instructions":
      st.write("## 📌 How to Use the System?")
      st.markdown("---")
    
      st.info("### 🖼 Step 1: Capture Your Image")
      st.success("Ensure your face is well-lit and free from obstructions like hair or glasses. The system will capture a clear image of your face for analysis.")

      st.info("### 🔍 Step 2: Facial Analysis")
      st.success("Once the image is captured, the system processes it to analyze different skin attributes such as oiliness, wrinkles, and pigmentation.")

      st.info("### 📊 Step 3: View Your Results")
      st.success("After processing, you will receive a detailed breakdown of your skin attributes, helping you understand areas that may need attention.")

      st.info("### ⚠️ Tips for Best Results")
      st.success(
        """
        - Use a well-lit environment to avoid shadows.
        - Ensure your face is clean and free from makeup.
        - Keep a neutral facial expression during image capture.
        - Avoid overexposure to bright lights that may affect skin detection.
        """
    )
    
    # Horizontal line
      st.markdown("---")
    
    # Footer
      st.caption("Developed by **Corbin Technology Solutions**")


    

    # Section 4

    elif selected_section == "👨‍💻 Credits":
      st.header("👨‍💻 Credits")
      st.markdown("---")
      

     # Highlight the developer information
      st.subheader("🌟 Developed By")
      st.markdown("""
     **Corbin Technology Solutions**  
     Delivering innovative technology solutions with a focus on Image Processing, and digital transformation.
     """)

     # Technologies used
      st.subheader("🛠️ Technologies Used")
      st.markdown("""
     - **Image Processing Techniques**: For analyzing facial attributes such as wrinkles, oiliness, and pigmentation.
     - **OpenCV**: For efficient facial image preprocessing and enhancement.
     - **Streamlit**: For building an interactive and user-friendly web interface.
     """)
    
      # Acknowledgment
      st.subheader("🙏 Acknowledgment")
      st.info("""
     Special thanks to the **Corbin Technology Solutions** team for their dedication to developing this Skin Analysis System.
     """)

      # Footer Message
      st.success("Your feedback helps us improve! Thank you for using the Facial Skin Analysis System.")


if __name__ == "__main__":
    main()
