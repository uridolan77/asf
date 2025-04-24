from asf.bollm.backend.repositories.model_dao import ModelDAO

def check_models():
    dao = ModelDAO()
    models = dao.get_all_models()
    print(f"Models in database: {len(models)}")
    for model in models:
        print(f"Model: {model['model_id']}, Provider: {model['provider_id']}")

if __name__ == "__main__":
    check_models()
