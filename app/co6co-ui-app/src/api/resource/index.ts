const image_BASE_URL="/api/resource/img"
const video_BASE_URL="/api/resource/video"
const poster_BASE_URL="/api/resource/poster" // /w/h
const thumbnail_BASE_URL="/api/resource/thumbnail" 
const defautlWidth=200;
const defautlHeidth=113;

const baseURL=import.meta.env.VITE_BASE_URL;

export const get_img_url = (path:string):string => {
    return `${baseURL}${image_BASE_URL}?path=${path}`; 
}; 

export const get_video_url = (path:string):string => {
    return `${baseURL}${video_BASE_URL}?path=${path}`; 
}; 
export const get_thumbnail_url = (path:string,w:number=defautlWidth,h:number=defautlHeidth):string => {
    return `${baseURL}${thumbnail_BASE_URL}/${w}/${h}?path=${path}`
};  
export const get_poster_url = (path:string,w:number=defautlWidth,h:number=defautlHeidth):string => {
    return `${baseURL}${poster_BASE_URL}/${w}/${h}?path=${path}`;
    //request_resource_svc("/api/biz/process/poster?path="+address).then(res=>{ playerOptions.value[index].poster=res}).catch(e=>playerOptions.value[index].poster="" );   
}; 

 