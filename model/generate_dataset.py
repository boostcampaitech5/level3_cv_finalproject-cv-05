def generate_dataset(project):
    project.generate_version(settings={
        "augmentation": {
            # "bbblur": { "pixels": 1.5 },
            # "bbbrightness": { "brighten": True, "darken": False, "percent": 91 },
            # "bbcrop": { "min": 12, "max": 71 },
            # "bbexposure": { "percent": 30 },
            # "bbflip": { "horizontal": True, "vertical": False },
            # "bbnoise": { "percent": 50 },
            # "bbninety": { "clockwise": True, "counter-clockwise": False, "upside-down": False },
            # "bbrotate": { "degrees": 45 },
            # "bbshear": { "horizontal": 45, "vertical": 45 },
            # "blur": { "pixels": 1.5 },
            # "brightness": { "brighten": True, "darken": False, "percent": 91 },
            # "crop": { "min": 12, "max": 71 },
            # "cutout": { "count": 26, "percent": 71 },
            # "exposure": { "percent": 30 },
            # "flip": { "horizontal": True, "vertical": False },
            # "hue": { "degrees": 180 },
            # "image": { "versions": 32 },
            # "mosaic": True,
            # "ninety": { "clockwise": True, "counter-clockwise": False, "upside-down": False },
            # "noise": { "percent": 50 },
            # "rgrayscale": { "percent": 50 },
            # "rotate": { "degrees": 45 },
            # "saturation": { "percent": 50 },
            # "shear": { "horizontal": 45, "vertical": 45 }
        },
        "preprocessing": {
            "auto-orient": True,
            # "contrast": { "type": "Contrast Stretching" },
            # "filter-null": { "percent": 50 },
            # "grayscale": True,
            # "isolate": True,
            #"remap": { "4": "plant" },
            "resize": { "width": 640, "height": 640, "format": "Stretch to" },
            # "static-crop": { "x_min": 10, "x_max": 90, "y_min": 10, "y_max": 90 },
            # "tile": { "rows": 2, "columns": 2 }
        
        }
    })