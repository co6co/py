# -*- encoding:utf-8 -*-
import co6co.utils.http as http
import co6co.utils.log as log
class googleSearch:
    def __init__(self):
        self.urlFarmat="https://www.google.com.hk/search?q={0}&sca_esv=559674773&source=hp&ei=gCLnZNzNCIbk2roPqoSR-AM&iflsig=AD69kcEAAAAAZOcwkIGZFfrwft-Ki_Jk76Rd1GOnUaFF&ved=0ahUKEwjcjb-x_fSAAxUGslYBHSpCBD8Q4dUDCAo&uact=5&oq=12&gs_lp=Egdnd3Mtd2l6IgIxMkjXB1DABFiIBXABeACQAQCYAQCgAQCqAQC4AQPIAQD4AQGoAgA&sclient=gws-wiz"
    def search(self,key:str):
        try:
            log.info(self.urlFarmat.format(key))
            print(http.get(self.urlFarmat.format(key), 30, "127.0.0.1:9666").text)
        except Exception as e:
            log.err(e)
            
if __name__ == '__main__': 
    gs=googleSearch()
    gs.search("abd")
    