"""
Script for creating preprocessing script
"""
from configparser import ConfigParser

template_file = './Self-Supervised/templates/preprocess_template.sh'
config_file = './Self-Supervised/config.ini'


def load_params(config_filename):
    config = ConfigParser()
    config.read(config_filename)
    return config


def replace_templates(template_file, config_dict, output_dir):
    print(config_dict)
    with open(output_dir + 'preprocess.sh', 'w') as out_file:
        with open(template_file, 'r') as temp_file:
            text = temp_file.read()
            for key, val in config_dict.items():
                key = '$'+key
                print(key, val)
                text = text.replace(key, val)
            out_file.write(text)


def add_files(config):
    """
    Script for copying run template to subdirectories
    """
    # Collect all parameters into a dict
    run_dict = {}
    gen_params = config['General']
    lp_params = config['LocalPicker']

    run_dict['box_size'] = gen_params['box_size']
    run_dict['pixel'] = gen_params['pixel']
    run_dict['ptl_size'] = gen_params['ptl_size']
    run_dict['ptl_class'] = gen_params['ptl_class']
    run_dict['ptl_pixel'] = gen_params['ptl_pixel']

    run_dict['lp_bin_size'] = lp_params['lp_bin_size']
    #run_dict['lp_defocus'] = lp_params['lp_defocus']
    run_dict['lp_max_sigma'] = lp_params['lp_max_sigma']


    replace_templates(template_file, run_dict, output_dir='./')


if __name__ == '__main__':
    print('Reading params from', config_file)
    config = load_params(config_file)
    add_files(config)
