# LLM Gateway Configuration Guide

This guide explains how to configure the LLM Gateway securely, especially for handling API keys and other sensitive information.

## Configuration Files

The LLM Gateway uses two configuration files:

1. **`llm_gateway_config.yaml`**: The base configuration file that is committed to the repository. This file should **never** contain actual API keys or other sensitive information.

2. **`llm_gateway_config.local.yaml`**: A local configuration file that extends the base configuration with sensitive information like API keys. This file is ignored by git and should never be committed to the repository.

## Setting Up Your Local Configuration

1. Copy the template file to create your local configuration:

   ```bash
   cp llm_gateway_config.local.yaml.template llm_gateway_config.local.yaml
   ```

2. Edit the local configuration file to add your API keys:

   ```yaml
   additional_config:
     providers:
       openai_gpt4_default:
         connection_params:
           api_key: YOUR_OPENAI_API_KEY_HERE
   ```

## Using the Secret Manager

For better security, you can use the Secret Manager to store your API keys:

1. Import your API keys from the local configuration:

   ```bash
   python -m asf.medical.llm_gateway.manage_secrets import
   ```

2. Verify that your secrets were imported:

   ```bash
   python -m asf.medical.llm_gateway.manage_secrets list --mask
   ```

3. Once your secrets are stored in the Secret Manager, you can remove the API keys from your local configuration file and use secret references instead:

   ```yaml
   additional_config:
     providers:
       openai_gpt4_default:
         connection_params:
           api_key_secret: llm:openai_api_key
   ```

## Environment Variables

You can also use environment variables to provide API keys:

1. Set the environment variable:

   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here

   # Linux/macOS
   export OPENAI_API_KEY=your_api_key_here
   ```

2. Configure the provider to use the environment variable:

   ```yaml
   additional_config:
     providers:
       openai_gpt4_default:
         connection_params:
           api_key_env_var: OPENAI_API_KEY
   ```

## Security Best Practices

1. **Never commit API keys** or other sensitive information to the repository.
2. Use the **Secret Manager** for storing sensitive information.
3. Use **environment variables** as a fallback.
4. Regularly **rotate your API keys** for better security.
5. Use **different API keys** for development, testing, and production environments.

## Troubleshooting

If you encounter issues with the LLM Gateway, you can use the diagnostic tools to troubleshoot:

```bash
python -m asf.medical.llm_gateway.diagnostic
```

For more detailed testing of the OpenAI connection:

```bash
python -m asf.medical.llm_gateway.test_with_secrets
```
