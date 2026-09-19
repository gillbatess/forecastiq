# Auto-deploy to Hugging Face on every GitHub push

1. GitHub repo -> Settings -> Secrets and variables -> Actions -> **New repository secret**
   Name: `HF_TOKEN`, value: a Hugging Face token with write access.
2. Create `.github/workflows/deploy-hf.yml` with:

```yaml
name: Deploy to Hugging Face Space
on:
  push:
    branches: [main]
  workflow_dispatch:
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -U huggingface_hub
      - run: python scripts/deploy_hf.py
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE: gillbatess/forecastiq
```

3. Commit and push. Check the Actions tab, then the Space's Logs tab.
