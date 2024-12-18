import { defineComponent, ref, reactive, onMounted, onBeforeUnmount, VNode } from 'vue'
import { Dialog, DialogInstance, byte2Unit } from 'co6co'

import { ElButton, ElMessageBox, ElCard, ElContainer, ElScrollbar } from 'element-plus'
import { upload_svc, get_upload_chunks_svc } from '@/api/file'

export default defineComponent({
  name: 'ModifyTask',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 110
    }
  },
  emits: {
    //@ts-ignore
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogRef = ref<DialogInstance>()
    const DATA = reactive<{
      files: Array<{ file: File; percentage?: number; subPath?: String; finshed?: boolean }>
      uploadFolder: String
    }>({
      files: [],
      uploadFolder: '/'
    })

    const readFileOrDirectory = (entry) => {
      //entry: FileEntry | DirectoryEntry
      if (entry.isFile) {
        entry.file((file: File) => {
          let subPath = (entry.fullPath as string).replace('/' + file.name, '')
          if (subPath) subPath = subPath.substring(1)
          DATA.files.push({ file: file, subPath: subPath })
        })
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader()
        dirReader.readEntries((entries) => {
          entries.forEach((entry) => readFileOrDirectory(entry))
        })
      }
    }
    const onDragOver = (event) => {
      // 阻止默认行为，以允许文件被放置到目标区域
      event.preventDefault()
    }
    const onDrop = (event: DragEvent) => {
      const items = event.dataTransfer?.items
      if (items && items.length > 0) {
        for (let i = 0; i < items.length; i++) {
          //webkitGetAsEntry 是一个非标准的方法
          const item = items[i].webkitGetAsEntry()
          if (item) {
            readFileOrDirectory(item)
          } else {
            // 如果不是文件系统条目（可能是普通文件）
            const file = event.dataTransfer?.files[i]
            if (file) DATA.files.push({ file: file })
          }
        }
      } else {
        // 如果 dataTransfer.items 为空，尝试从 dataTransfer.files 获取文件
        const files = event.dataTransfer?.files
        if (files && files.length > 0) {
          DATA.files = Array.from(files).map((f) => {
            return { file: f }
          })
        }
      }
      // 阻止默认行为，以防止浏览器打开文件
      event.preventDefault()
    }
    const upload = async () => {
      for (let i = 0; i < DATA.files.length; i++) {
        const opt = DATA.files[i]
        if (opt.finshed) continue
        const file = opt.file
        const chunks = createFileChunks(file)
        const uploadedChunks = await getUploadedChunks(file.name, chunks.length)

        // 过滤掉已经上传的块
        const remainingIndex: Array<number> = []
        const remainingChunks = chunks
          .map((v, index) => ({ index: index + 1, value: v }))
          .filter((v) => !uploadedChunks.includes(v.index))
        if (remainingChunks.length === 0) {
          //console.log('所有块已上传完毕')
          opt.percentage = 100
          continue
        }
        console.info('上传：', remainingChunks, remainingIndex)
        opt.finshed = await uploadFileChunks(
          remainingChunks,
          opt.subPath as string,
          file.name,
          chunks.length,
          (p: number) => {
            opt.percentage = p
          }
        )
      }
      const unfinshed = DATA.files.filter((o) => !o.finshed)
      if (unfinshed.length > 0) return false
      return true
    }
    const onUpload = async () => {
      if (!DATA.files || DATA.files.length == 0) {
        ElMessageBox.alert('请选择要上传的文件或文件夹！')
      }
      await upload()
    }
    /** 分片上传 */
    const createFileChunks = (file: File, chunkSize = 1 * 1024 * 1024) => {
      const chunks: Array<Blob> = []
      let start = 0
      while (start < file.size) {
        const end = Math.min(file.size, start + chunkSize)
        const chunk = file.slice(start, end)
        chunks.push(chunk)
        start = end
      }
      return chunks
    }
    const getUploadPath = (root: string, subPath: string) => {
      if (!root) root = '/'
      if ((root.endsWith('/') || root.endsWith('\\')) && subPath) return root + subPath
      else if (subPath) return root + '/' + subPath
      return root
    }
    const uploadFileChunks = async (
      chunks: [{ index: number; value: Blob }],
      subPath: string,
      fileName: string,
      totalChunks: number,
      bck: (p: number) => void
    ) => {
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i]
        const formData = new FormData()
        formData.append('file', chunk.value, `${fileName}_part${chunk.index}`)
        formData.append('index', chunk.index.toString())
        formData.append('totalChunks', totalChunks.toString())
        formData.append('fileName', fileName)
        formData.append('uploadPath', getUploadPath(DATA.uploadFolder as string, subPath)!)
        try {
          await upload_svc(formData)
          bck((totalChunks - chunks.length + i + 1) / totalChunks)
          //console.log(`块 ${i + 1} 上传成功`, response.message)
        } catch (error) {
          console.error(`块 ${i + 1} 上传失败:`, error)
          // 处理错误，例如重试或提示用户
          return false
        }
      }
      return true
    }
    const getUploadedChunks = async (fileName: string, totalChunks: number) => {
      const response = await get_upload_chunks_svc({
        fileName: fileName,
        totalChunks: totalChunks,
        uploadPath: DATA.uploadFolder as string
      })
      return response.data.uploadedChunks || []
    }

    /** end 分片上传 */
    const fromSlots = {
      buttons: () => (
        <>
          <ElButton type="warning" onClick={onUpload} v-slots={{ default: () => '开始上传' }} />
        </>
      ),
      default: () => (
        <>
          <ElCard
            v-slots={{
              header: () => <ElButton>选择上传</ElButton>
            }}
          >
            <ElScrollbar onDrop={onDrop} onDragover={onDragOver}>
              {DATA.files.length == 0 ? (
                <p>拖动文件或文件夹到这里</p>
              ) : (
                <ul>
                  {DATA.files.map((opt) => (
                    <li>
                      {opt.file.name} {byte2Unit(opt.file.size, 'b', 2)}
                      {new Date(opt.file.lastModified).toISOString()}
                      {opt.percentage ? (opt.percentage * 100).toFixed(2) + '%' : ''}
                    </li>
                  ))}
                </ul>
              )}
            </ElScrollbar>
          </ElCard>
        </>
      )
    }

    const rander = (): VNode => {
      return (
        <Dialog
          closeTxt="取消上传"
          title={prop.title}
          style={ctx.attrs}
          ref={diaglogRef}
          v-slots={fromSlots}
        />
      )
    }
    const openDialog = (folder: string) => {
      DATA.uploadFolder = folder
      DATA.files = []
      diaglogRef.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
