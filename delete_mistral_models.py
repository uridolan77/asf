from asf.bollm.backend.repositories.model_dao import ModelDAO

def delete_mistral_models():
    dao = ModelDAO()
    
    # Delete mistral-large-latest
    deleted = dao.delete_model('mistral-large-latest', 'mistral')
    if deleted:
        print("Deleted model mistral-large-latest")
    else:
        print("Failed to delete model mistral-large-latest")
    
    # Delete mistral-medium-latest
    deleted = dao.delete_model('mistral-medium-latest', 'mistral')
    if deleted:
        print("Deleted model mistral-medium-latest")
    else:
        print("Failed to delete model mistral-medium-latest")
    
    # Check remaining models
    models = dao.get_all_models()
    print(f"Remaining models in database: {len(models)}")
    for model in models:
        print(f"Model: {model['model_id']}, Provider: {model['provider_id']}")

if __name__ == "__main__":
    delete_mistral_models()
