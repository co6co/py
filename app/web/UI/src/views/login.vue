<template>
  <div class="login-wrap">
    <div class="ms-login">
      <div class="ms-title">{{ systeminfo.name }}</div>
      <el-form :model="DATA" :rules="rules" ref="login" label-width="0px" class="ms-content">
        <el-form-item prop="username">
          <el-input v-model="DATA.username" placeholder="用户名">
            <template #prepend>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            type="password"
            placeholder="密 码"
            v-model="DATA.password"
            @keyup.enter="submitForm(login)"
          >
            <template #prepend>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item prop="verify">
          <DragVerify :height="30" v-model="DATA.verify" :onVerifySuccess="onVerifySuccess" />
        </el-form-item>
        <div class="login-btn">
          <el-button type="primary" @click="submitForm(login)">登录</el-button>
        </div>
        <p class="login-tips">Tips : {{ message }}</p>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { Lock, User } from '@element-plus/icons-vue'

import { isDebug } from '../utils'
import { useTagsStore } from '../store/tags'
import { userSvc, registerRoute, DragVerify } from 'co6co-right'
import { storeAuthonInfo, showLoading, closeLoading, isMobileBrowser } from 'co6co'

import { router, ViewObjects } from '../router'
import type { FormInstance, FormRules } from 'element-plus'
import useSystem from '../hooks/useSystem'
import { getPublicURL } from '../utils'

interface LoginInfo {
  username: string
  password: string
  verify: string
}
let message = ref('')
const DATA = reactive<LoginInfo>({
  username: '',
  password: '',
  verify: ''
})
if (isDebug) {
  DATA.username = 'admin'
  DATA.password = 'admin12345'
}
const onVerifySuccess = (_) => {
  login?.value?.validateField('verify')
}
const rules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  verify: [
    {
      required: true,
      message: '请拖动滑块验证' + isMobileBrowser(),
      trigger: 'blur',
      validator(rule, value, callback, source, options) {
        if (DATA.verify) {
          callback()
        } else {
          const msg = rule.message
          callback(new Error(typeof msg == 'function' ? msg() : msg))
        }
      }
    }
  ]
}

const { systeminfo } = useSystem()
const login = ref<FormInstance>()
const submitForm = (formEl: FormInstance | undefined) => {
  if (!formEl) return
  formEl.validate((valid: boolean) => {
    if (valid) {
      showLoading()
      userSvc
        .login_svc(
          { userName: DATA.username, password: DATA.password, verifyCode: DATA.verify },
          5000,
          false
        )
        .then((res) => {
          message.value = res.message

          storeAuthonInfo(res.data, DATA.username)
          registerRoute(ViewObjects, router, () => {
            window.location.href = getPublicURL('/')
          })
        })
        .catch((e) => {
          DATA.verify = ''
          ElMessage.error(`登录出错：${e.message}`)
        })
        .finally(() => {
          closeLoading()
        })
    } else {
      message.value = '数据验证失败'
      return Promise.reject('valid Form Error')
    }
  })
}

const tags = useTagsStore()
tags.clearTags()
</script>

<style scoped>
.login-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  background-image: url(../assets/img/login-bg_01.jpg);
  background-size: 100%;
}
.ms-title {
  width: 100%;
  line-height: 50px;
  text-align: center;
  font-size: 20px;
  color: #fff;
  border-bottom: 1px solid #ddd;
}
.ms-login {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 350px;
  margin: -190px 0 0 -175px;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.3);
  overflow: hidden;
}
.ms-content {
  padding: 30px 30px;
}
.login-btn {
  text-align: center;
}
.login-btn button {
  width: 100%;
  height: 36px;
  margin-bottom: 10px;
}
.login-tips {
  font-size: 12px;
  line-height: 30px;
  color: red;
}
</style>
