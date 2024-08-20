import base64
import urllib
import requests
import json

API_KEY = "********"
SECRET_KEY = "********"

def main():
        
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b?access_token=" + get_access_token()
    

    payload = json.dumps({
        "prompt": "Please carefully observe this picture and complete the following tasks:  1. Describe all the objects contained in this image and list their features and positions as detailed as possible. Please include the color, shape, size, and relative position of the object (e.g. upper left corner, middle, lower right corner).",
        #"temperature": 0.65,
        #"penalty_score": 1.0,
        "image": get_file_content_as_base64("*******#文件路径",False)
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
    

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()


