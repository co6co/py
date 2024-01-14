import {number2hex} from '../../../utils'
import * as p from '../../../components/stream/src/types/ptz';
import { ptz } from '..';
export  const ptzCmd=(sip:string,speed:number,type:p.ptz_type, name: p.ptz_name)=>{
    let ptzCmd: number[] = [0xA5, 0x0F, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
    if (type == 'starting') { 
        switch (name) {
            case 'up':
                ptzCmd[3] = 0x08;
                ptzCmd[5] = speed & 0xFF; 
                break;
            case 'down':
                ptzCmd[3] = 0x04;
                ptzCmd[5] = speed & 0xFF; 
                break;
            case 'right':
                ptzCmd[3] = 0x01;
                ptzCmd[4] = speed & 0xFF; 
                break;
            case 'left':
                    ptzCmd[3] = 0x02;
                    ptzCmd[4] = speed & 0xFF; 
                break;
            case 'zoomin':
            ptzCmd[3] = 0x44;
            ptzCmd[5] = speed & 0xFF; 
                break;
            case 'zoomout':
                ptzCmd[3] = 0x48;
                ptzCmd[5] = (-speed) & 0xFF; 
                break;
        }
    } else {
         //0
    }
    let check:number = 0;
    let cmdstr:string="";
    let tmp:number[ ]=[0,0,0,0,0,0,0,0]
     
    for (let i = 0; i < 7; i++)
    {
        console.info( i,'==>',ptzCmd[i])
        check  += ptzCmd[i]; 
        cmdstr += number2hex(ptzCmd[i],2)  
    }
    ptzCmd[7] = check % 256; 
    let ttt=""
    for (let i = 0; i < 8; i++)
    {
        ttt+=`${number2hex(ptzCmd[i],2)} `
        
    }
    console.error(ttt)

    console.info( 7,'==>',ptzCmd[7])
    cmdstr += number2hex(ptzCmd[7],2)  ;  
    console.info(cmdstr)
    let sn = new Date().getMilliseconds();
    let xml = `
    <?xml version="1.0" encoding="UTF-8"?>
    <Control>
        <CmdType>DeviceControl</CmdType>
        <SN>${sn}</SN>
        <DeviceID>${sip}</DeviceID>
        <PTZCmd>${cmdstr}</PTZCmd>
    </Control> 
    `;
    return xml
}



export  const ptzCmdStr=(ptzCmd:string)=>{
    if (ptzCmd&&ptzCmd.length!=16) console.error("命令不符规格")
    console.info("data:")
    let arr:number[]=[0,0,0,0,0,0,0,0]
    for(let i=0,j=0;i<ptzCmd.length;i+=2){
        console.info(`${ptzCmd[i]}${ptzCmd[i+1]}`)
        arr[j]=parseInt(`${ptzCmd[i]}${ptzCmd[i+1]}`,16)
    }
    let d4=arr[3]
    let zoomOut=32
    let zoomInt=16
    let up=8
    let down=4
    let left=2
    let right

    let d5=arr[4] //水平速度
    let d6=arr[5] //垂直速度

    //Iris
    let fiIriBig=72 //FI 光圈小
    let fiIriSmail=68 //FI 光圈大
    //Focus
    let FocusIn=66 //近
    let FocusOut=65 //远
    d5=arr[4] //聚焦速度
    d6=arr[4] //光圈速度 
}
