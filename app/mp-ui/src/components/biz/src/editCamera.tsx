import { defineComponent, ref, reactive, provide } from 'vue'
import type { InjectionKey } from 'vue'
import { default as EcDiaglogForm } from '../../common/EcDiaglogForm'

import { FormOperation } from '../../common/types'
import type { ObjectType, FormData } from '../../common/types'

import * as api from '../../../api/site/ipCamera'
import {
  ElRow,
  ElCol,
  ElButton,
  ElFormItem,
  ElInput,
  ElMessage,
  ElMessageBox,
  ElContainer,
  ElHeader,
  ElMain,
  type FormRules,
  type FormInstance,
  ElCard
} from 'element-plus'
import { Plus, Minus } from '@element-plus/icons-vue'

//Omit、Pick、Partial、Required
export type FormItem = Omit<api.Item, 'id' | 'createTime' | 'updateTime'> & {
  streamUrls: Array<StreamURlOption>
}

export interface StreamURlOption {
  name: String
  url: String
}
export default defineComponent({
  name: 'diaglogForm',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 90
    }
  },
  emits: {
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<InstanceType<typeof EcDiaglogForm>>()
    const data = reactive<FormData<number, FormItem>>({
      operation: FormOperation.add,
      id: 0,
      fromData: {
        innerIp: '',
        name: '',
        sip: '',
        code: '',
        uuid: '',
        ip: '',
        siteId: -1,
        channel1_sip: '',
        channel2_sip: '',
        channel3_sip: '',
        channel4_sip: '',
        channel5_sip: '',
        channel6_sip: '',
        streams: '',
        streamUrls: [],
        ptzTopic: '/MANSCDP_cmd'
      }
    })
    const key = Symbol('formData') as InjectionKey<FormItem> //'formData'
    provide('formData', data.fromData)
    provide('title', prop.title)

    const init_data = (oper: FormOperation, siteId: number, item?: api.Item) => {
      data.operation = oper

      switch (oper) {
        case FormOperation.add:
          data.id = -1
          data.fromData.innerIp = ''
          data.fromData.name = ''
          data.fromData.sip = ''
          data.fromData.channel1_sip = ''
          data.fromData.channel2_sip = ''
          data.fromData.channel3_sip = ''
          data.fromData.channel4_sip = ''
          data.fromData.channel5_sip = ''
          data.fromData.siteId = siteId
          data.fromData.ptzTopic = '/MANSCDP_cmd'
          data.fromData.streamUrls = []
          break
        case FormOperation.edit:
          if (!item) return false
          data.id = item.id
          data.fromData.uuid = item.uuid
          data.fromData.innerIp = item.innerIp
          data.fromData.name = item.name
          data.fromData.sip = item.sip
          data.fromData.channel1_sip = item.channel1_sip
          data.fromData.channel2_sip = item.channel2_sip
          data.fromData.channel3_sip = item.channel3_sip
          data.fromData.channel4_sip = item.channel4_sip
          data.fromData.channel5_sip = item.channel5_sip
          data.fromData.ptzTopic = item.ptzTopic
          data.fromData.siteId = item.siteId
          if (item.streams && typeof item.streams == 'string')
            data.fromData.streamUrls = JSON.parse(item.streams)
          else data.fromData.streamUrls = []
          break
      }
      return true
    }

    const rules: FormRules = {
      siteId: [{ required: true, message: '请选择所属站点', trigger: ['blur'] }],
      name: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
      sip: [{ required: true, message: '请输入sip地址', trigger: 'blur' }],
      channel1_sip: [{ required: true, len: 20, message: '请输入通道2地址', trigger: 'blur' }],
      channel2_sip: [{ required: true, len: 20, message: '请输入通道2地址', trigger: 'blur' }],
      channel3_sip: [{ len: 20, message: '请输入通道3地址', trigger: 'blur' }],

      innerIp: [{ required: true, message: '请输入设备IP', trigger: ['blur'] }],
      streamName: [{ required: true, message: '请视频地址名称', trigger: ['blur'] }],
      streamUrl: [{ required: true, message: '请视频地址', trigger: ['blur'] }],
      ptzTopic:[
        { required: true,    message: '请视频地址', trigger: ['blur'] },
        { pattern: /^((\/)[a-zA-z0-9-_.]{1,}){1,}$/, message: '请输入主题名,必须以 / 开头', trigger: 'blur' }
 
      ]
    }
    const save = () => {
      //提交数据
      let promist: Promise<IPageResponse<any>>
      switch (data.operation) {
        case FormOperation.add:
          promist = api.add_svc(data.fromData)
          break
        case FormOperation.edit:
          promist = api.edit_svc(data.id, data.fromData)
          break
        default:
          return
      }
      promist.then((res) => {
        if (res.code == 0) {
          diaglogForm.value?.closeDialog()
          ElMessage.success(`操作成功`)
          ctx.emit('saved', res.data)
        } else {
          ElMessage.error(`操作失败:${res.message}`)
        }
      })
    }

    const removeConfig = (index: number, item: Partial<StreamURlOption>) => {
      data.fromData.streamUrls.splice(index, 1)
    }
    const addConfig = () => {
      data.fromData.streamUrls.push({ name: '', url: '' })
    }
    //通过通道号生成流地址
    const onGenerateStreamAddress = () => {
      ElMessageBox.confirm(`将删除以前的流地址信息，确定要生成？`, '提示', {
        type: 'warning'
      })
        .then(() => {
          let streamUrl = []
          let channelArr = []
          channelArr.push(data.fromData.channel1_sip)
          channelArr.push(data.fromData.channel2_sip)
          channelArr.push(data.fromData.channel3_sip)
          channelArr.push(data.fromData.channel4_sip)
          channelArr.push(data.fromData.channel5_sip)
          channelArr.push(data.fromData.channel6_sip)
          for (let i = 0; i < channelArr.length; i++) {
            let value = channelArr[i]
            if (value && value.length == 20) {
              let t = {
                name: '通道' + (i + 1),
                url: `wss://stream.jshwx.com.cn:8441/flv_ws?device=gb${value}&type=rt.flv`
              }
              streamUrl.push(t)
            }
          }
          if (streamUrl.length == 0) {
            ElMessage.warning('请先输入对应的通道号！')
          }
          data.fromData.streamUrls = streamUrl
        })
        .catch(() => {})
    }
    const fromSlots = {
      buttons: () => (
        <>
          <ElButton
            onClick={() => {
              diaglogForm.value?.validate(save)
            }}
          >
            保存
          </ElButton>
        </>
      ),
      default: () => (
        <>
          <ElFormItem label="设备UUID" prop="uuid">
            <ElInput readonly={true} v-model={data.fromData.uuid}></ElInput>
          </ElFormItem>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="名称" prop="name">
                <ElInput v-model={data.fromData.name}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="设备编码" prop="code">
                <ElInput v-model={data.fromData.code}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="内网IP" prop="innerIp">
                <ElInput v-model={data.fromData.innerIp}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="IP地址" prop="ip">
                <ElInput v-model={data.fromData.ip}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElFormItem label="云台控制主题" prop="ptzTopic">
            <ElInput v-model={data.fromData.ptzTopic}></ElInput>
          </ElFormItem>
          <ElFormItem label="SIP地址" prop="sip">
            <ElInput v-model={data.fromData.sip}></ElInput>
          </ElFormItem>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="通道1SIP地址" prop="channel1_sip">
                <ElInput v-model={data.fromData.channel1_sip}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="通道2SIP地址" prop="channel2_sip">
                <ElInput v-model={data.fromData.channel2_sip}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>

          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="通道3SIP地址" prop="channel3_sip">
                <ElInput v-model={data.fromData.channel3_sip}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="通道4SIP地址" prop="channel4_sip">
                <ElInput v-model={data.fromData.channel4_sip}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElRow>
            <ElCol span={12}>
              <ElFormItem label="通道5SIP地址" prop="channel5_sip">
                <ElInput v-model={data.fromData.channel5_sip}></ElInput>
              </ElFormItem>
            </ElCol>
            <ElCol span={12}>
              <ElFormItem label="通道6SIP地址" prop="channel6_sip">
                <ElInput v-model={data.fromData.channel6_sip}></ElInput>
              </ElFormItem>
            </ElCol>
          </ElRow>
          <ElContainer>
            <ElMain style="padding:0">
              {data.fromData.streamUrls.map((item, index) => {
                return (
                  <ElRow>
                    <ElCol span={9}>
                      <ElFormItem
                        label={`流${index + 1}名称`}
                        prop={'streamUrls.' + index + '.name'}
                        rules={{
                          required: true,
                          message: `流${index + 1}名称`,
                          trigger: 'blur'
                        }}
                      >
                        <ElInput v-model={item.name} placeholder="视频流名称"></ElInput>
                      </ElFormItem>
                    </ElCol>
                    <ElCol span={13}>
                      <ElFormItem
                        label={`流${index + 1}地址`}
                        prop={'streamUrls.' + index + '.url'}
                        rules={{
                          required: true,
                          message: `流${index + 1}地址`,
                          trigger: 'blur'
                        }}
                      >
                        <ElInput v-model={item.url} placeholder="配置内容"></ElInput>
                      </ElFormItem>
                    </ElCol>
                    <ElCol span={2}>
                      <ElButton
                        onClick={() => {
                          removeConfig(index, item)
                        }}
                        icon={Minus}
                      ></ElButton>
                    </ElCol>
                  </ElRow>
                )
              })}
              <ElRow>
                <ElCol span={16}></ElCol>
                <ElCol span={6} style="text-align: right;">
                  {' '}
                  <ElButton onClick={onGenerateStreamAddress}>通过SIP地址生成流地址</ElButton>{' '}
                </ElCol>
                <ElCol span={2}>
                  <ElButton onClick={addConfig} icon={Plus}></ElButton>
                </ElCol>
              </ElRow>
            </ElMain>
          </ElContainer>
        </>
      )
    }
    const openDialog = (siteId: number, ipCameraId?: number) => {
      let oper = FormOperation.add
      if (ipCameraId) {
        //编辑
        api.get_svc(ipCameraId).then((res) => {
          if (res.data) {
            oper = FormOperation.edit
            init_data(oper, siteId, res.data)
            diaglogForm.value?.openDialog()
          } else {
            ElMessage.error('未能找到对应的监控球机,请刷新重试！')
          }
        })
      } else {
        init_data(oper, siteId)
        diaglogForm.value?.openDialog()
      }
    }

    const rander = (): ObjectType => {
      return (
        <EcDiaglogForm
          title={prop.title}
          labelWidth={prop.labelWidth}
          style={ctx.attrs}
          rules={rules}
          ref={diaglogForm}
          v-slots={fromSlots}
        ></EcDiaglogForm>
      )
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
