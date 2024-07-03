export interface resourceOption{
    url:string, 
    poster:string,
    type: 0|1 ,//0 video 1：image
    name:string,
} 
 
export interface imageOption{
    url:string,  
    thumbnail:string,
    name?:string,
} 
 
export interface videoOption{
    url:string, 
    poster:string,  
} 
 