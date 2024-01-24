import {types} from '../../../devices'
export interface player {
    //dom?: any;
    url: string;
    data?: types.DeviceData
    streamList: Array<types.Stream>;
}
export interface PlayerList {
    splitNum: number;
    isFullScreen: boolean;
    currentWin: number;
    currentStreams: Array<types.Stream>;
    players: Array<player>;
}


