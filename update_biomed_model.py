from asf.bollm.backend.repositories.model_dao import ModelDAO

def update_model(old_model_id, new_model_id, provider_id):
    """Update a model to a new version.
    
    Args:
        old_model_id: The ID of the old model to replace
        new_model_id: The ID of the new model to create
        provider_id: The provider ID for both models
    """
    dao = ModelDAO()
    
    # Get the old model
    old_model = dao.get_model_by_id(old_model_id, provider_id)
    if not old_model:
        print(f"Model {old_model_id} not found for provider {provider_id}")
        return
    
    print(f"Found model: {old_model}")
    
    # Delete the old model
    deleted = dao.delete_model(old_model_id, provider_id)
    if not deleted:
        print(f"Failed to delete model {old_model_id}")
        return
    
    print(f"Deleted model {old_model_id}")
    
    # Create the new model
    new_model = {
        "model_id": new_model_id,
        "provider_id": provider_id,
        "display_name": old_model.get("display_name", f"{new_model_id}"),
        "model_type": old_model.get("model_type", "chat"),
        "context_window": old_model.get("context_window", 16000),
        "max_output_tokens": old_model.get("max_output_tokens", 2000),
        "capabilities": old_model.get("capabilities", []),
        "parameters": old_model.get("parameters", {})
    }
    
    created_model = dao.create_model(new_model)
    if not created_model:
        print(f"Failed to create model {new_model_id}")
        return
    
    print(f"Created model: {created_model}")

if __name__ == "__main__":
    # Example usage:
    # Replace with your actual model IDs and provider
    update_model('old-model-id', 'new-model-id', 'provider-id')
