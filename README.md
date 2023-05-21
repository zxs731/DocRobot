# DocRobot

#本地部署 
1 pip install -r requirements.txt
2 set environment var in your local system
  # Set this to `azure`
  export OPENAI_API_TYPE=azure
  # The API version you want to use: set this to `2022-12-01` for the released version.
  export OPENAI_API_VERSION=2022-12-01
  # The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
  export OPENAI_API_BASE=https://your-resource-name.openai.azure.com
  # The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
  export OPENAI_API_KEY=<your Azure OpenAI API key>
3 flask run
4 open 127.0.0.1:5000 with your browser
5 upload a txt file then start to talk with bot
  ![Uploading screen.png…]()
