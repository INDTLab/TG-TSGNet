from omegaconf import OmegaConf
config_path = "/data2/yifan/taming-transformers-master/logs/open_images_distilled_e_60/configs/2021-09-03T13-00-00-project.yaml"
config = OmegaConf.load(config_path)
import yaml
print(yaml.dump(OmegaConf.to_container(config)))
from taming.models.cond_transformer import Net2NetTransformer
model = Net2NetTransformer(**config.model.params)