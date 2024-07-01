from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):

    def preprocess(self, requests):
    	```
    	requests 받은 데이터의 전처리
    	```
    	pass
    
    def inference(self, x):
    	```
        .preprocess에서 받은 데이터로 인퍼런스
        ```
        pass
        
    def postprocess(self, preds):
    	```
        .inference에서 받은 데이터로 결과 후처리 후
        array로 반환
        ```
        return ['ok']

