import os
import json
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import gaussian
from skimage.transform import resize
class SEN12FloodDataset:
    def __init__(self, img_size=256, max_samples=None):
        self.root_dir = "/opt/airflow/data/"
        self.img_size = img_size
        self.max_samples = max_samples
        self.required_bands = ['b02', 'b03', 'b04', 'b08']
        
        s1_json_path = os.path.join(self.root_dir, "S1list.json")
        s2_json_path = os.path.join(self.root_dir, "S2list.json")
        with open(s1_json_path) as f:
            self.s1_metadata = json.load(f)
        with open(s2_json_path) as f:
            self.s2_metadata = json.load(f)
        
        self.samples = self._prepare_samples()
        print(f"Found {len(self.samples)} valid S1+S2 sample pairs.")

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
                                   
                        s2_base = s2_info['filename']
                        for band in self.required_bands:
                            sample[band] = os.path.join(scene_dir, f"{s2_base}_{band}.tif")
                        
                        if self._validate_sample(sample):
                            samples.append(sample)
        
        return samples

    def _validate_sample(self, sample):
        try:        
            if not all(os.path.exists(v) for k, v in sample.items() if k not in ['flooding', 'date', 'scene_id']):
                
                print(os.path.exists(os.path.exists(v) for k, v in sample.items() if k not in ['flooding', 'date', 'scene_id']))
                return False
            with rasterio.open(sample['s1_vv']) as src:
                vv_shape = src.shape
            with rasterio.open(sample['b02']) as src:
                b02_shape = src.shape
                
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
            vv = self._load_and_preprocess(sample['s1_vv'], sar=True)
            vh = self._load_and_preprocess(sample['s1_vh'], sar=True)
            s2_bands = [self._load_and_preprocess(sample[band], sar=False) for band in self.required_bands]
            
            image = np.concatenate([
                np.stack([vv, vh]),
                np.stack(s2_bands)
            ], axis=0)
            
            return {
                'image': image,
                'label': int(sample['flooding']),
                'date': sample['date'],
                'scene_id': sample['scene_id']
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return self._create_dummy_sample()

    def _load_and_preprocess(self, path, sar=False):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

        if sar:
            data = 10 * np.log10(data + 1e-6)
            data = gaussian(data, sigma=1, preserve_range=True)

        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-6)
        data = resize(data, (self.img_size, self.img_size), order=1, preserve_range=True, anti_aliasing=True)
        return data

    def _create_dummy_sample(self):
        return {
            'image': np.zeros((len(self.required_bands) + 2, self.img_size, self.img_size)),
            'label': -1,
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
            if sample['label'] == -1:
                continue
            try:
                features.append(self._extract_features_from_sample(sample))
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
                
        return pd.DataFrame(features) if features else pd.DataFrame()

    def _extract_features_from_sample(self, sample):
        image = sample['image']
        
        features = {
            'date': sample['date'],
            'scene_id': sample['scene_id'],
            'label': sample['label']
        }
        
        vv, vh = image[0], image[1]
        features.update({
            'vv_mean': vv.mean(),
            'vh_mean': vh.mean(),
            'vv_vh_ratio': (vv / (vh + 1e-6)).mean(),
            'vv_std': vv.std(),
            'vh_std': vh.std(),
        })
        
        features.update(self._calc_texture_features(vv, 'vv_'))
        features.update(self._calc_texture_features(vh, 'vh_'))
        
        b03, b04, b08 = image[2], image[3], image[4]
        features.update({
            'ndwi': ((b03 - b08) / (b03 + b08 + 1e-6)).mean(),
            'ndvi': ((b08 - b04) / (b08 + b04 + 1e-6)).mean(),
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
    
    features_df.to_csv(os.path.join(output_dir, "flood_features.csv"), index=False)
    
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(features_df.corr(), annot=False, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title("Feature Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.png"))
        plt.close()
        print("Feature analysis saved.")
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

# Example usage
if __name__ == "__main__":
    dataset = SEN12FloodDataset(img_size=128, max_samples=100)
    extractor = FloodFeatureExtractor(dataset)
    features_df = extractor.extract_features(num_samples=50)
    analyze_features(features_df)
