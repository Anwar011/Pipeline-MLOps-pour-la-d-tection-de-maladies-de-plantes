try:
    from transformers import ViTModel, ViTConfig
    print('Import successful')
except ImportError:
    try:
        from transformers.models.vit import ViTModel, ViTConfig
        print('Alternative import successful')
    except ImportError as e:
        print(f'Import failed: {e}')
