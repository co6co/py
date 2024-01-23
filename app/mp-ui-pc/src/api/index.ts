import axios  from '../utils/request';
import  {ResponseType,Method} from 'axios';
import  {AxiosInstance, AxiosError, AxiosResponse, AxiosRequestConfig} from 'axios';
 
//创建 Blob 资源
export const create_URL_resource=(resource:{data:Blob }):string=>{
    return URL.createObjectURL(resource.data) 
}
//下载  blob 资源 
export const download_blob_resource=(resource:{data:Blob,fileName:string})=>{
  const link = document.createElement('a')
  link.href = create_URL_resource({data:resource.data}) 
  link.download = resource.fileName
  link.click()
  window.URL.revokeObjectURL(link.href) 
}
// 请求文件资源
//todo mp 端 没有 baseURL:""
export const request_resource_svc=async (url:string,axios_config:AxiosRequestConfig={method:"get", responseType :"blob"}) =>{ 
    let default_config:{method:Method, url:string,  baseURL:"",   timeout:number }={
        method:'get',//请求方式
        url:url,//请求地址  会加上 baseURL  
        timeout:30000,
        baseURL:"" 
    } 
    const res = await axios({... default_config, ...axios_config})   
    //const blob = new Blob([res.data]);//处理文档流  
    const result=create_URL_resource({data:res.data}) 
    return result
    //request_resource_svc("/api/xxxx/poster?path="+address).then(res=>{ option.value.poster=res}).catch(e=>option.value.poster="" );   
} 

//下载文件
//download_config 为默认时获取文件长度
export const download_fragment_svc=(url:string,config:download_config={method:"HEAD"})=>{ 
    let default_config:{method:Method,baseURL:string,url:string,responseType:ResponseType,timeout:number }={
        method:'get',//请求方式
        url:url,//请求地址  会加上 baseURL
        responseType :"blob",//文件流将会被转成blob 
        baseURL:"",
        timeout:30000 
    }
    //Object.assign({},default_config,config) 
    return axios({... default_config, ...config}  ) 
} 


//单独下载
export const download_svc= (url:string,fileName:string)=>{
    axios({
        method:'get',//请求方式
        url:url,//请求地址  会加上 baseURL
        responseType:'blob',//文件流将会被转成blob
        baseURL:"",
        timeout:3*60000  
    }).then(res => {
        try{
            //console.info(res,res)
            const blob = new Blob([res.data]);//处理文档流 
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
