import os
import json
import rasterio
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
import seaborn as sns

class SEN12FloodDataset(Dataset):
    def __init__(self, img_size=256, max_samples=None):
        self.root_dir = "/opt/airflow/data/SEN12FLOOD/"
        self.img_size = img_size
        self.max_samples = max_samples
        self.required_bands = ['b02', 'b03', 'b04', 'b08']  # Required bands (excluding optional b11)
        
        s1_json_path = "/opt/airflow/data/SEN12FLOOD/S1list.json"
        s2_json_path = "/opt/airflow/data/SEN12FLOOD/S2list.json"
        # Load metadata
        with open(s1_json_path) as f:
            self.s1_metadata = json.load(f)
        with open(s2_json_path) as f:
            self.s2_metadata = json.load(f)
        
        # Prepare samples with strict validation
        self.samples = self._prepare_samples()
        print(f"Found {len(self.samples)} valid S1+S2 sample pairs after validation")

    def _prepare_samples(self):
        samples = []
        s2_scenes = {k: v for k, v in self.s2_metadata.items() if k.isdigit()}
        
        for scene_id, scene_data in tqdm(self.s1_metadata.items(), desc="Processing scenes"):
            if not scene_id.isdigit() or scene_id not in s2_scenes:
                continue
                
            scene_dir = os.path.join(self.root_dir, scene_data['folder'])
            if not os.path.exists(scene_dir):
                continue
                
            s1_images = {k: v for k, v in scene_data.items() if k.isdigit()}
            s2_images = {k: v for k, v in self.s2_metadata[scene_id].items() if k.isdigit()}
            
            for s1_img_id, s1_info in s1_images.items():
                for s2_img_id, s2_info in s2_images.items():
                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples
                    
                    s1_date = self._normalize_date(s1_info['date'])
                    s2_date = self._normalize_date(s2_info['date'])
                    
                    if s1_date == s2_date:
                        sample = {
                            's1_vv': self._find_s1_file(scene_dir, s1_info['filename'], 'VV'),
                            's1_vh': self._find_s1_file(scene_dir, s1_info['filename'], 'VH'),
                            'flooding': s1_info['FLOODING'],
                            'date': s1_date,
                            'scene_id': scene_id
                        }
                        
                        # Add required bands
                        s2_base = s2_info['filename']
                        for band in self.required_bands:
                            sample[band] = os.path.join(scene_dir, f"{s2_base}_{band}.tif")
                        
                        # Verify all required files exist and have valid shapes
                        if self._validate_sample(sample):
                            samples.append(sample)
        
        return samples

    def _validate_sample(self, sample):
        """Validate that all files exist and have compatible dimensions"""
        try:
            # Check file existence
            if not all(v and os.path.exists(v) for k, v in sample.items() 
                      if k not in ['flooding', 'date', 'scene_id']):
                return False
            
            # Check initial dimensions
            with rasterio.open(sample['s1_vv']) as src:
                vv_shape = src.shape
            with rasterio.open(sample['b02']) as src:
                b02_shape = src.shape
                
            # Ensure all bands have same shape as b02
            for band in self.required_bands[1:]:
                with rasterio.open(sample[band]) as src:
                    if src.shape != b02_shape:
                        return False
            
            return True
        except:
            return False

    def _find_s1_file(self, scene_dir, base_name, pol):
        patterns = [
            f"{base_name}_corrected_{pol}.tif",
            f"{base_name}_{pol}.tif",
            f"{base_name}_{pol}.tiff",
            f"{base_name}_{pol.lower()}.tif"
        ]
        for pattern in patterns:
            path = os.path.join(scene_dir, pattern)
            if os.path.exists(path):
                return path
        return None

    def _normalize_date(self, date_str):
        try:
            if '-' in date_str:
                return date_str
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            return self._create_dummy_sample()
            
        sample = self.samples[idx]
        
        try:
            # Load SAR data with shape validation
            vv = self._load_and_preprocess(sample['s1_vv'], sar=True)
            vh = self._load_and_preprocess(sample['s1_vh'], sar=True)
            
            # Load optical data
            s2_bands = []
            for band in self.required_bands:
                band_data = self._load_and_preprocess(sample[band], sar=False)
                s2_bands.append(band_data)
            
            # Stack all bands (2 SAR + 4 optical)
            image = np.concatenate([
                np.stack([vv, vh], axis=0),
                np.stack(s2_bands, axis=0)
            ], axis=0)
            
            return {
                'image': torch.tensor(image, dtype=torch.float32),
                'label': torch.tensor(int(sample['flooding']), dtype=torch.long),
                'date': sample['date'],
                'scene_id': sample['scene_id']
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return self._create_dummy_sample()

    def _load_and_preprocess(self, path, sar=False):
        """Load and preprocess single band"""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
        
        if sar:
            data = 10 * np.log10(data + 1e-6)
            data = cv2.GaussianBlur(data, (3, 3), 0)
        
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-6)
        data = cv2.resize(data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return data

    def _create_dummy_sample(self):
        """Create a dummy sample when loading fails"""
        return {
            'image': torch.zeros((len(self.required_bands) + 2, self.img_size, self.img_size)),
            'label': torch.tensor(-1, dtype=torch.long),
            'date': '',
            'scene_id': ''
        }

class FloodFeatureExtractor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.texture_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    def extract_features(self, num_samples=None):
        features = []
        num_samples = min(num_samples or len(self.dataset), len(self.dataset))
        
        for i in tqdm(range(num_samples), desc="Extracting features"):
            sample = self.dataset[i]
            if sample['label'].item() == -1:
                continue
                
            try:
                features.append(self._extract_features_from_sample(sample))
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
                
        return pd.DataFrame(features) if features else pd.DataFrame()

    def _extract_features_from_sample(self, sample):
        image = sample['image'].numpy()
        
        # Basic metadata
        features = {
            'date': sample['date'],
            'scene_id': sample['scene_id'],
            'label': sample['label'].item()
        }
        
        # SAR features
        vv = image[0]
        vh = image[1]
        features.update({
            'vv_mean': vv.mean(),
            'vh_mean': vh.mean(),
            'vv_vh_ratio': (vv / (vh + 1e-6)).mean(),
            'vv_std': vv.std(),
            'vh_std': vh.std(),
        })
        
        # Texture features
        features.update(self._calc_texture_features(vv, 'vv_'))
        features.update(self._calc_texture_features(vh, 'vh_'))
        
        # Spectral indices
        b03 = image[2]  # Green (index 2 because SAR bands are first)
        b04 = image[3]  # Red
        b08 = image[4]  # NIR
        
        features.update({
            'ndwi': (b03 - b08) / (b03 + b08 + 1e-6).mean(),
            'ndvi': (b08 - b04) / (b08 + b04 + 1e-6).mean(),
        })
        
        return features
    
    def _calc_texture_features(self, img, prefix):
        img_8bit = ((img - img.min()) * (255 / (img.max() - img.min() + 1e-6))).astype(np.uint8)
        glcm = graycomatrix(img_8bit, distances=[1], angles=[0], symmetric=True, normed=True)
        return {
            f'{prefix}{prop}': graycoprops(glcm, prop)[0, 0]
            for prop in self.texture_props
        }

def analyze_features(features_df, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    if features_df.empty:
        print("No valid features to analyze")
        return None
    
    # Save raw features
    features_df.to_csv(os.path.join(output_dir, "flood_features.csv"), index=False)
    
    try:
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        if 'label' not in numeric_cols:
            print("No label column found for correlation analysis")
            return None
            
        correlations = features_df[numeric_cols].corr()['label'].drop('label', errors='ignore')
        corr_df = pd.DataFrame({'feature': correlations.index, 'correlation': correlations.values})
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df.sort_values('abs_correlation', ascending=False, inplace=True)
        corr_df.to_csv(os.path.join(output_dir, "feature_correlations.csv"), index=False)
        
        # Plot top correlations
        plt.figure(figsize=(12, 8))
        sns.barplot(x='abs_correlation', y='feature', data=corr_df.head(20))
        plt.title("Top Feature Correlations with Flood Labels")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlations.png"))
        plt.close()
        
        return corr_df
    except Exception as e:
        print(f"Error during feature analysis: {str(e)}")
        return None

def visualize_sample(sample, output_path=None):
    image = sample['image']
    
    plt.figure(figsize=(18, 12))
    
    # SAR bands
    plt.subplot(2, 3, 1)
    plt.imshow(image[0], cmap='gray')
    plt.title(f"VV Band\nDate: {sample['date']}\nFlood: {sample['label'].item()}")
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(image[1], cmap='gray')
    plt.title("VH Band")
    plt.colorbar()
    
    # False color composite
    plt.subplot(2, 3, 3)
    plt.imshow(np.stack([image[4], image[3], image[2]], axis=-1))  # NIR, Red, Green
    plt.title("False Color (NIR-Red-Green)")
    
    # NDVI
    plt.subplot(2, 3, 4)
    ndvi = (image[4] - image[3]) / (image[4] + image[3] + 1e-6)  # (NIR-Red)/(NIR+Red)
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("NDVI")
    plt.colorbar()
    
    # NDWI
    plt.subplot(2, 3, 5)
    ndwi = (image[2] - image[4]) / (image[2] + image[4] + 1e-6)  # (Green-NIR)/(Green+NIR)
    plt.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.title("NDWI")
    plt.colorbar()
    
    # VV/VH ratio
    plt.subplot(2, 3, 6)
    ratio = image[0] / (image[1] + 1e-6)
    plt.imshow(ratio, cmap='viridis')
    plt.title("VV/VH Ratio")
    plt.colorbar()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    try:
        # Configuration
     
        
       
        dataset = SEN12FloodDataset()
        print(dataset)
        if len(dataset) > 0:
            print(f"Successfully loaded {len(dataset)} samples")
            
            # Visualize first valid sample
            valid_samples = [s for s in dataset.samples if os.path.exists(s['s1_vv'])]
            if valid_samples:
                sample_idx = dataset.samples.index(valid_samples[0])
                sample = dataset[sample_idx]
                visualize_sample(sample, "results/sample_visualization.png")
            else:
                print("No valid samples found for visualization")
            
            # Extract features
            extractor = FloodFeatureExtractor(dataset)
            features_df = extractor.extract_features()
            
            if not features_df.empty:
                # Analyze features
                correlations = analyze_features(features_df)
                if correlations is not None:
                    print("\nTop feature correlations:")
                    print(correlations.head(10))
            else:
                print("No valid features extracted")
        else:
            print("No valid samples found in dataset")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")