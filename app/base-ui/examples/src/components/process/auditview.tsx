import { defineComponent, computed, ref, type PropType } from 'vue'
import * as pd from '../../api/pd'

import {
  ElButtonGroup,
  ElEmpty,
  ElButton,
  ElTooltip,
  ElRow,
  ElCol,
  ElScrollbar,
  ElDescriptions,
  ElDescriptionsItem,
  ElIcon,
  ElTag,
  ElText,
  ElSpace,
  ElSelect,
  ElOption,
  ElMessage
} from 'element-plus'
import { Timer, Location, Bell, Position, Memo, HelpFilled, EditPen } from '@element-plus/icons-vue'
import { htmlPlayer as HtmlPlayer, types as playerType } from '../player'
import Images from '../player/Images'
import * as r from '../../api/resource'
import imgStyle from '../../assets/css/images.module.less'
import '../../assets/css/auditview.less'
import useLabelSelect from '../../hook/useLabelSelect'
import { download_one_svc } from '../../api/process'
import { usePermission, ViewFeature } from '../../hook/sys/useRoute'
export enum AuditType {
  error,
  succ,
  alarm,
  ignore
}
export interface IAuditResult {
  type: AuditType
  label?: string
}
export const getAuditTypeDesc = (type?: AuditType) => {
  switch (type) {
    case AuditType.error:
      return '误警'
    case AuditType.succ:
      return '确警'
    case AuditType.alarm:
      return '通过[下发]'
    case AuditType.ignore:
      return '忽略'
    default:
      return '未审核'
  }
}

export const getTagType = (audit: number) => {
  const auditType = audit
  switch (auditType) {
    case AuditType.error:
      return 'danger'
    case AuditType.succ:
      return ''
    case AuditType.alarm:
      return 'success'
    case AuditType.ignore:
      return 'warning'
    default:
      return 'info'
  }
}
export default defineComponent({
  name: 'AuditView',
  props: {
    disabledAudit: {
      type: Boolean,
      default: false
    },
    record: {
      type: Object as PropType<pd.Iorders_record>
      // required: true
    }
  },
  emits: {
    audited: (data: IAuditResult) => true
  },
  setup(props, context) {
    const { getPermissKey } = usePermission()
    const allowAudit = computed(() => {
      return props.record != undefined
    })
    const { selectData } = useLabelSelect()
    //const labelRef=ref("")
    const onAuditSumit = (auditType: AuditType) => {
      context.emit('audited', { type: auditType, label: props.record?.label })
    }
    const downloading = ref(false)
    const onDownload = () => {
      if (props.record) {
        downloading.value = true
        download_one_svc(
          props.record.id,
          { boatName: props.record.boat_name, vioName: props.record.vio_name },
          () => {
            downloading.value = false
          }
        )
      } else ElMessage.error('数据未加载不能下载')
    }

    //图片地址
    const imageUrls = computed(() => {
      if (!props.record) return []
      const urls: Array<string> = []
      if (props.record.pic1_save_path) urls.push(r.get_img_url(props.record.pic1_save_path))
      if (props.record.anno_pic1_save_path)
        urls.push(r.get_img_url(props.record.anno_pic1_save_path))
      return urls
    })
    //视频地址
    const videoOption = computed<playerType.videoOption>(() => {
      if (!props.record) return { url: '', poster: '' }
      return {
        url: r.get_video_url(props.record.video_save_path),
        poster: r.get_poster_url(props.record.video_save_path)
      }
    })
    return () => {
      return (
        <>
          <ElScrollbar>
            <ElRow>
              <ElCol style="height: 100%; overflow: auto">
                <ElRow>
                  <ElCol span={12}>
                    <Images list={imageUrls.value}></Images>
                  </ElCol>
                  <ElCol span={12}>
                    <div class={imgStyle.imageList}>
                      <HtmlPlayer option={videoOption.value}></HtmlPlayer>
                    </div>
                  </ElCol>
                </ElRow>
                <ElRow style="padding:5px;min-height:145px">
                  <ElCol span={12}>
                    {props.record ? (
                      <ElDescriptions column={2} title="告警信息" class="recordDescs">
                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <Position />
                                  </ElIcon>
                                  记录ID：
                                </ElText>
                              )
                            }
                          }}
                        >
                          {props.record?.id}
                        </ElDescriptionsItem>
                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <HelpFilled />
                                  </ElIcon>
                                  AI审核结果：
                                </ElText>
                              )
                            }
                          }}
                        >
                          <ElTag type={getTagType(props.record?.program_audit_result)}>
                            {getAuditTypeDesc(props.record?.program_audit_result)}
                          </ElTag>
                        </ElDescriptionsItem>
                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <Timer />
                                  </ElIcon>
                                  告警时间：
                                </ElText>
                              )
                            }
                          }}
                        >
                          {props.record?.dev_record_time}
                        </ElDescriptionsItem>
                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <Location />
                                  </ElIcon>
                                  船名：
                                </ElText>
                              )
                            }
                          }}
                        >
                          {props.record?.boat_name}
                        </ElDescriptionsItem>
                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <Bell />
                                  </ElIcon>
                                  违反规则：
                                </ElText>
                              )
                            }
                          }}
                        >
                          {props.record?.vio_name}
                        </ElDescriptionsItem>

                        <ElDescriptionsItem
                          v-slots={{
                            label: () => {
                              return (
                                <ElText>
                                  <ElIcon>
                                    <EditPen />
                                  </ElIcon>
                                  标签：
                                </ElText>
                              )
                            }
                          }}
                        >
                          {props.disabledAudit ? (
                            <>{props.record?.label}</>
                          ) : (
                            <div class="flex flex-wrap" style="display:inline-block">
                              <ElSelect
                                size="small"
                                clearable
                                style="width:120px"
                                v-model={props.record.label}
                                placeholder="请选择标签"
                              >
                                {selectData.value.map((d, index) => {
                                  return (
                                    <ElOption key={index} label={d.alias} value={d.name}></ElOption>
                                  )
                                })}
                              </ElSelect>
                            </div>
                          )}
                        </ElDescriptionsItem>
                      </ElDescriptions>
                    ) : (
                      <ElEmpty description="未加载数据" style="padding:0" imageSize={40}></ElEmpty>
                    )}
                  </ElCol>
                  <ElCol span={12}>
                    {props.disabledAudit ? (
                      <>{context.slots.default ? context.slots.default() : <div></div>}</>
                    ) : (
                      <>
                        <ElButtonGroup class="auditButtonGroup">
                          <ElButton
                            v-permiss={getPermissKey(ViewFeature.download)}
                            loading={downloading.value}
                            onClick={() => onDownload()}
                            disabled={!allowAudit.value}
                            icon="Download"
                          >
                            记录下载
                          </ElButton>

                          <ElTooltip content="使用 ‘1’ 快捷键">
                            <ElButton
                              onClick={() => onAuditSumit(AuditType.error)}
                              type="danger"
                              disabled={!allowAudit.value}
                              icon="WarningFilled"
                            >
                              {getAuditTypeDesc(AuditType.error)}
                            </ElButton>
                          </ElTooltip>
                          <ElTooltip content="使用 ‘2’ 快捷键">
                            <ElButton
                              onClick={() => onAuditSumit(AuditType.succ)}
                              type="primary"
                              disabled={!allowAudit.value}
                              icon="Check"
                            >
                              {getAuditTypeDesc(AuditType.succ)}
                            </ElButton>
                          </ElTooltip>
                          <ElTooltip content="使用 ‘3’ 快捷键">
                            <ElButton
                              onClick={() => onAuditSumit(AuditType.alarm)}
                              type="success"
                              disabled={!allowAudit.value}
                              icon="UploadFilled"
                            >
                              {getAuditTypeDesc(AuditType.alarm)}
                            </ElButton>
                          </ElTooltip>
                          <ElTooltip content="使用 ‘4’ 快捷键">
                            <ElButton
                              onClick={() => onAuditSumit(AuditType.ignore)}
                              type="warning"
                              disabled={!allowAudit.value}
                              icon="Notebook"
                            >
                              {getAuditTypeDesc(AuditType.ignore)}
                            </ElButton>
                          </ElTooltip>
                        </ElButtonGroup>
                        {context.slots.default ? context.slots.default() : <></>}
                      </>
                    )}
                  </ElCol>
                </ElRow>
              </ElCol>
            </ElRow>
          </ElScrollbar>
        </>
      )
    }
  }
})
