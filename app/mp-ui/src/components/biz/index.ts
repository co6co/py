import bizPlayer from './src/bizPlayer.vue'
import * as types from './src/types'
import editIpCamera from './src/editIpCamera.vue'
import taklerPtz from './src/taklerPtz.vue'
import * as ptz from './src/ptz'
import { types as pType } from '../player'

interface AlarmItem {
  id: number
  uuid: string
  alarmType: string
  videoUid: string
  rawImageUid: string
  markedImageUid: string
  alarmTime: string
  createTime: string
}
const getResourceUrl = (uuid: string, isposter: boolean = false) => {
  if (isposter) return import.meta.env.VITE_BASE_URL + `/api/resource/poster/${uuid}/700/600`
  return import.meta.env.VITE_BASE_URL + `/api/resource/${uuid}`
}
const getResources = (item: AlarmItem) => {
  let data: Array<pType.resourceOption> = []
  if (item.rawImageUid) {
    data.push({
      url: getResourceUrl(item.rawImageUid),
      name: '原始图片',
      poster: getResourceUrl(item.rawImageUid, true),
      type: 1
    })
  }
  if (item.markedImageUid) {
    data.push({
      url: getResourceUrl(item.markedImageUid),
      name: '标注图片',
      poster: getResourceUrl(item.markedImageUid, true),
      type: 1
    })
  }
  if (item.videoUid) {
    data.push({
      url: getResourceUrl(item.videoUid),
      name: '原始视频',
      poster: getResourceUrl(item.videoUid, true),
      type: 0
    })
  }
  return data
}
export { bizPlayer, types, editIpCamera, taklerPtz, ptz, type AlarmItem, getResources }
