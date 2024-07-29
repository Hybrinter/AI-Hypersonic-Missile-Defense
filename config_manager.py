import json
import os


class ConfigManager:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.configs = {}

    def load_config(self, config_name):
        with open(os.path.join(self.config_dir, f"{config_name}.json"), 'r') as file:
            self.configs[config_name] = json.load(file)

    def load_configs(self):
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]
                self.load_config(config_name)

    def get_config(self, config_name):
        return self.configs.get(config_name, None)
