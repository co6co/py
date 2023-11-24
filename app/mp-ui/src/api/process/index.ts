import { config } from 'md-editor-v3/lib/MdEditor/config';
import axios, {AxiosInstance, AxiosError, AxiosResponse, AxiosRequestConfig} from 'axios';
import request  from '../../utils/request'; 
const BASE_URL="/api/biz/process"
export const get_status_svc = (): Promise<IPageResponse> => {
    return request.get(`${BASE_URL}/getStatus`,{params:{noLogin: true}});
}; 

export const queryList_svc = (data:any): Promise<IPageResponse> => {
    return request.post(`${BASE_URL}/list`,data,{data:data});
}; 
export const audit_svc = (data:any): Promise<IResponse> => {
    return request.post(`${BASE_URL}/audit`, data);
}; 
export const position_svc = (data:any): Promise<IResponse> => {  
    return request.post(`${BASE_URL}/position`, data, {params:{noLogin: true}});
}; 
export const one_svc = (id:number): Promise<IResponse> => {   
    return request.get(`${BASE_URL}/one/${id}`,  {params:{noLogin: true}});
}; 

export const  start_download_task= (data:any): Promise<IResponse> => {   
    return request.post(`${BASE_URL}/startDownloadTask`,data);
};   
//单独下载
export const download_one_svc= (id:number,data:{boatName:string,vioName:string})=>{
    request({
        method:'get',//请求方式
        url:`${BASE_URL}/download/${id}`,//请求地址
        responseType:'blob'//文件流将会被转成blob  
    }).then(res => {
        try{
            //console.info(res,res)
            const blob = new Blob([res.data]);//处理文档流
            const fileName = `${data.boatName}_${data. vioName}_${id}.zip`;
            const down = document.createElement('a');
            down.download = fileName;
            down.style.display = 'none';//隐藏,没必要展示出来
            down.href = URL.createObjectURL(blob);
            document.body.appendChild(down);
            down.click();
            URL.revokeObjectURL(down.href); // 释放URL 对象
            document.body.removeChild(down);//下载完成移除 
        }catch(e){
            console.error(e)
        }
    })   
}
 
export const  get_log_content_svc= (id:number): Promise<AxiosResponse> => {   
    return request.post(`${BASE_URL}/auditLog/${id}`,null,{responseEncoding:"utf-8",responseType:"text"});
};   