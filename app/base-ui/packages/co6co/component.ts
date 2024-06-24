//import { EcDiaglogDetail } from '@co6co/components/tsx'

import { ElAlert } from '@co6co/components/alert'
import { EcDialog } from '@co6co/components/dialog'
import { EcDetail } from '@co6co/components/detail'
import { EcDialogDetail } from '@co6co/components/diaglogDetail'
import { EcForm } from '@co6co/components/form'
import { EcDialogForm } from '@co6co/components/dialogForm'
import { closeLoading, showLoading } from '@co6co/components/glogining'
import { EcIconSelect } from '@co6co/components/iconSelect'
import { ElIcon } from '@co6co/components/icon'
import type { Plugin } from 'vue'
export default [
  ElAlert,
  ElIcon,
  EcDialog,
  EcDetail,
  EcDialogDetail,
  EcForm,
  EcDialogForm,
  EcIconSelect,
  closeLoading,
  showLoading,
] as Plugin[]
