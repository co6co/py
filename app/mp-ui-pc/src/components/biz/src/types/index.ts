import {types} from '../../../devices'
export interface player {
    dom?: any;
    url: string;
    data?: types.deviceItem
    streamList: Array<stream_source>;
}
export interface PlayerList {
    splitNum: number;
    isFullScreen: boolean;
    currentWin: number;
    currentStreams: Array<stream_source>;
    players: Array<player>;
}


