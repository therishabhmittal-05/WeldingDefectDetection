import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="X-Ray Welding Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class ImageProcessor:
    """Class to handle various image processing techniques"""
    
    @staticmethod
    def fourier_transform(image, cutoff_freq=0.1):
        """Apply Fourier Transform filtering"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask with cutoff frequency
        mask = np.zeros((rows, cols), np.uint8)
        radius = int(cutoff_freq * min(rows, cols))
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
        
        # Apply mask and inverse FFT
        f_shift_masked = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize to 0-255
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return img_back
    
    @staticmethod
    def gaussian_blur(image, kernel_size=5):
        """Apply Gaussian blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def edge_enhancement(image, strength=1.0):
        """Enhance edges using unsharp masking technique"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        
        return unsharp_mask
    
    @staticmethod
    def adjust_hsv_v(image, value_shift=0):
        """Adjust HSV V (brightness/value) component"""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] + value_shift
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale images, treat as brightness adjustment
            return np.clip(image.astype(np.float32) + value_shift, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_hsv_s(image, saturation_shift=0):
        """Adjust HSV S (saturation) component"""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] + saturation_shift
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale images, return unchanged
            return image
    
    @staticmethod
    def adjust_hsv_h(image, hue_shift=0):
        """Adjust HSV H (hue) component"""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale images, return unchanged
            return image
    
    @staticmethod
    def adjust_contrast(image, contrast=1.0):
        """Adjust contrast"""
        return np.clip(contrast * image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if len(image.shape) == 3:
            # Convert to LAB color space and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            return clahe.apply(image)
    
    @staticmethod
    def unsharp_masking(image, radius=1.0, amount=1.0, threshold=0):
        """Apply unsharp masking"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), radius)
        
        # Create the unsharp mask
        unsharp = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply threshold
        unsharp = np.where(np.abs(unsharp) < threshold, 0, unsharp)
        
        # Apply the mask
        result = gray.astype(np.float32) + amount * unsharp
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def highpass_filter(image, cutoff=0.1):
        """Apply high-pass filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask that blocks low frequencies
        mask = np.ones((rows, cols), np.uint8)
        radius = int(cutoff * min(rows, cols))
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
        
        # Apply mask and inverse FFT
        f_shift_masked = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_masked)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return img_back
    
    @staticmethod   
    def invert_colors(img):
            
        return 255 - img
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)
    
    # @staticmethod
    # def lowpass_filter(image, cutoff=0.3):
    #     """Apply low-pass filter"""
    #     if len(image.shape) == 3:
    #         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray = image.copy()
        
    #     # Apply FFT
    #     f_transform = np.fft.fft2(gray)
    #     f_shift = np.fft.fftshift(f_transform)
        
    #     # Create low-pass mask
    #     rows, cols = gray.shape
    #     crow, ccol = rows // 2, cols // 2
        
    #     # Create a mask that allows low frequencies
    #     mask = np.zeros((rows, cols), np.uint8)
    #     radius = int(cutoff * min(rows, cols))
    #     cv2.circle(mask, (ccol, crow), radius, 1, -1)
        
    #     # Apply mask and inverse FFT
    #     f_shift_masked = f_shift * mask
    #     f_ishift = np.fft.ifftshift(f_shift_masked)
    #     img_back = np.fft.ifft2(f_ishift)
    #     img_back = np.abs(img_back)
        
    #     # Normalize
    #     img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    #     return img_back

# class Inference:
#     def __init__(self, model):
#         self.model = model

#     def predict(self, image):
#         results = self.model(image)
#         return results

def main():
    
    # print("Entered main function") #####

    st.title("X-Ray Welding Defect Augmentation")
    st.markdown("---")

    # model = load_model(model_name)
    # Initialize processor and detector
    processor = ImageProcessor()
    # detector = Inference(model)


    # print("Model loaded") #####


    # File uploader
    uploaded_file = st.file_uploader(
        "Upload X-Ray Image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an X-ray image for defect detection and augmentation analysis"
    )
    # print("File loader")
    if uploaded_file is not None:

        # print("File Uploaded") #####

        # Load original image
        original_image = Image.open(uploaded_file)
        original_array = np.array(original_image)
        
        # Create two main sections
        st.markdown("## Original Image Analysis")
        # col1, col2 = st.columns([1, 1])

        # with col1: 
        st.markdown("---")        
        st.subheader("Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])  # Centered layout
        with col2:
            st.image(original_array, caption="Original X-Ray Image")

        # st.image(original_array, caption="Original X-Ray Image", use_container_width=True)
            
        # print("Column 2") #####
        # with col2: 
        #     # Display predictions if available
            
        #     st.markdown("---")
        #     st.subheader("Model Inference (Original Image)")

        #     if st.button("🚀 Run Inference (Original)", type="primary", use_container_width=True):
        #         print("I am here")
        #         with st.spinner("Running inference on original image..."):
        #             print("About to run prediction...")
        #             original_predictions = detector.predict(original_image)
        #             print("Prediction done")

        #             st.session_state.original_predictions = original_predictions

        #         if 'original_predictions' in st.session_state:
        #             try:
        #                 result = st.session_state.original_predictions[0]
        #                 plotted_array = result.plot()
        #                 st.image(Image.fromarray(plotted_array), caption="Original Image Predictions", use_container_width=True)                    
        #             except Exception as e:
        #                 st.error(f"Failed to display prediction: {e}")

        # print("Above this is Original Image and Model run function") #####
        

        st.markdown("## Image Processing & Analysis")
        col1, col2 = st.columns([1, 1])

  
        with col1:
        # Process image with selected augmentations
            processed_image = original_array.copy()

            st.markdown("---")
            # Image processing controls
            st.subheader("Image Processing Controls")


            apply_invert = st.checkbox("Invert Colors (X-Ray)")

            # CLAHE
            apply_clahe = st.checkbox("CLAHE")
            if apply_clahe:
                clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 0.0, 0.5)
                # clahe_clip_manual = st.number_input("Manual CLAHE Clip", 1.0, 10.0, clahe_clip, 0.1, format="%.1f")
                # clahe_clip = clahe_clip_manual
                
                clahe_grid = st.slider("CLAHE Grid Size", 4, 16, 8, 2)
                # clahe_grid_manual = st.number_input("Manual CLAHE Grid", 4, 16, clahe_grid, 1)
                # clahe_grid = clahe_grid_manual


            # Gamma Correction
            apply_gamma = st.checkbox("Gamma Correction")
            if apply_gamma:
                gamma_value = st.slider("Gamma Value", 0.1, 3.0, 1.0, 0.1, help="Lower than 1.0 brightens the image; higher darkens it")

            # Gaussian Blur
            apply_blur = st.checkbox("Gaussian Blur")
            if apply_blur:
                # blur_kernel = st.slider("Blur Kernel Size", 1, 21, 5, 1)
                blur_manual = st.number_input("Blur Kernel Size", 1, 21, 5, 1)
                blur_kernel = blur_manual if blur_manual % 2 == 1 else blur_manual + 1
            
            

            


            # HSV V (Value/Brightness)
            apply_hsv_v = st.checkbox("HSV-V (Brightness)")
            if apply_hsv_v:
                hsv_v_val = st.slider("HSV-V Shift", 0.0, 1.0, 0.0, 0.005)
                # hsv_v_manual = st.number_input("Manual HSV-V Shift", 0.0, 1.0, hsv_v_val, 0.005)
                # hsv_v_val = hsv_v_manual
            
            # HSV S (Saturation)
            apply_hsv_s = st.checkbox("HSV-S (Saturation)")
            if apply_hsv_s:
                hsv_s_val = st.slider("HSV-S Shift", 0.0, 1.0, 0.0, 0.005)
                # hsv_s_manual = st.number_input("Manual HSV-S Shift", 0.0, 1.0, hsv_s_val, 0.005)
                # hsv_s_val = hsv_s_manual

            # HSV H (Hue)
            apply_hsv_h = st.checkbox("HSV-H (Hue)")
            if apply_hsv_h:
                hsv_h_val = st.slider("HSV-H Shift", 0.0, 1.0, 0.0, 0.005)
                # hsv_h_manual = st.number_input("Manual HSV-H Shift", 0.0, 1.0, hsv_h_val, 0.005)
                # hsv_h_val = hsv_h_manual
            
            # Edge Enhancement
            apply_edge = st.checkbox("Edge Enhancement")
            if apply_edge:
                edge_strength = st.slider("Edge Enhancement Strength", 0.1, 3.0, 1.0, 0.1)
                edge_manual = st.number_input("Manual Edge Strength", 0.1, 3.0, edge_strength, 0.01, format="%.2f")
                edge_strength = edge_manual

            # Unsharp Masking
            apply_unsharp = st.checkbox("Unsharp Masking")
            if apply_unsharp:
                unsharp_radius = st.slider("Unsharp Radius", 0.5, 5.0, 1.0, 0.1)
                unsharp_radius_manual = st.number_input("Manual Unsharp Radius", 0.5, 5.0, unsharp_radius, 0.01, format="%.2f")
                unsharp_radius = unsharp_radius_manual
                
                unsharp_amount = st.slider("Unsharp Amount", 0.1, 3.0, 1.0, 0.1)
                unsharp_amount_manual = st.number_input("Manual Unsharp Amount", 0.1, 3.0, unsharp_amount, 0.01, format="%.2f")
                unsharp_amount = unsharp_amount_manual    
            
            # Fourier Transform
            apply_fourier = st.checkbox("Fourier Transform (Low pass Filtering)")
            if apply_fourier:
                fourier_cutoff = st.slider("Fourier Cutoff Frequency", 0.01, 0.5, 0.1, 0.01, 
                                        help="Lower values allow fewer low frequencies")
                fourier_manual = st.number_input("Manual Fourier Cutoff", 0.01, 0.5, fourier_cutoff, 0.001, format="%.3f")
                fourier_cutoff = fourier_manual
            
            
            # High-pass Filter
            apply_highpass = st.checkbox("High-pass Filter")
            if apply_highpass:
                highpass_cutoff = st.slider("High-pass Cutoff", 0.01, 0.3, 0.1, 0.01)
                highpass_manual = st.number_input("Manual High-pass Cutoff", 0.01, 0.3, highpass_cutoff, 0.001, format="%.3f")
                highpass_cutoff = highpass_manual

            
            
            
            
            
            # Apply augmentations in sequence
            if apply_invert:
                processed_image = processor.invert_colors(processed_image)

            if apply_clahe:
                processed_image = processor.apply_clahe(processed_image, clahe_clip, int(clahe_grid))


            if apply_gamma:
                processed_image = processor.adjust_gamma(processed_image, gamma_value)


            if apply_fourier:
                processed_image = processor.fourier_transform(processed_image, fourier_cutoff)
            
            if apply_blur:
                processed_image = processor.gaussian_blur(processed_image, int(blur_kernel))
            
            if apply_edge:
                processed_image = processor.edge_enhancement(processed_image, edge_strength)
            
            if apply_hsv_v:
                processed_image = processor.adjust_hsv_v(processed_image, hsv_v_val)
            
            if apply_hsv_s:
                processed_image = processor.adjust_hsv_s(processed_image, hsv_s_val)
            
            if apply_hsv_h:
                processed_image = processor.adjust_hsv_h(processed_image, hsv_h_val)
            
            
            
            if apply_unsharp:
                processed_image = processor.unsharp_masking(processed_image, unsharp_radius, unsharp_amount)
            
            if apply_highpass:
                processed_image = processor.highpass_filter(processed_image, highpass_cutoff)
            
            

            
        with col2: 
            st.markdown("---")        
            st.subheader("Processed Image")
            if len(processed_image.shape) == 2:
                st.image(processed_image, caption="Processed X-Ray Image", use_container_width=True, clamp=True)
            else:
                st.image(processed_image, caption="Processed X-Ray Image", use_container_width=True)

        # # Display processed image and inference section
        #     st.markdown("---")        

        #     st.subheader("Model Inference")

        #     if st.button("🚀 Run Inference (Processed)", type="primary", use_container_width=True):
        #         with st.spinner("Running inference on processed image..."):
        #             processed_predictions = detector.predict(processed_image)
        #             st.session_state.processed_predictions = processed_predictions
            
        #     if 'processed_predictions' in st.session_state:
        #         try:
        #             result_pred = st.session_state.processed_predictions[0]
        #             plotted_array = result_pred.plot()
        #             st.image(Image.fromarray(plotted_array), caption="Processed Image Predictions", use_container_width=True)
        #         except Exception as e:
        #             st.error(f"Failed to display prediction: {e}")
            

    else:
        st.info("👆 Please upload an X-ray image to begin analysis")
        
        # Show sample instructions
        # st.markdown("""
        # ### 🚀 Getting Started
        
        # 1. **Upload an X-ray image** using the file uploader above
        # 2. **Section 1**: View original image and run inference to get baseline predictions
        # 3. **Section 2**: Apply various image processing techniques with interactive controls
        # 4. **Run inference** on processed image to compare results
        
        # ### 🔧 Available Image Processing Techniques
        
        # - **Fourier Transform**: Frequency domain filtering
        # - **Gaussian Blur**: Noise reduction and smoothing  
        # - **Edge Enhancement**: Sharpen structural details
        # - **HSV-V Adjustment**: Brightness/Value modification
        # - **HSV-S Adjustment**: Saturation modification
        # - **HSV-H Adjustment**: Hue modification
        # - **Contrast Adjustment**: Linear contrast modification
        # - **CLAHE**: Adaptive histogram equalization
        # - **Unsharp Masking**: Advanced edge sharpening
        # - **High-pass Filter**: Emphasize fine details
        # - **Low-pass Filter**: Remove high-frequency noise
        
        # ### 📝 Notes
        
        # - Each slider has a corresponding manual input field for precise control
        # - The defect detection model shown is a **mock implementation** for demonstration
        # - Replace the `MockDefectDetector` class with your actual trained model
        # - Process images in Section 2 and compare results with Section 1 predictions
        # """)

if __name__ == "__main__":
    main()
