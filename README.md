# DocRobot

#本地部署 
1 pip install -r requirements.txt
2 set environment var in your local system
  export OPENAI_API_TYPE=azure
  export OPENAI_API_VERSION=2022-12-01
  export OPENAI_API_BASE=https://your-resource-name.openai.azure.com
  export OPENAI_API_KEY=<your Azure OpenAI API key>
3 flask run
4 open 127.0.0.1:5000 with your browser
5 upload a txt file then start to talk with bot
  ![Uploading screen.png…]()
