<template>
  <div class="login-wrap">
    <div class="ms-login">
      <div class="ms-title">AI 数据审核系统</div>
      <el-form :model="param" :rules="rules" ref="login" label-width="0px" class="ms-content">
        <el-form-item prop="username">
          <el-input v-model="param.username" placeholder="用户名">
            <template #prepend>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            type="password"
            placeholder="密 码"
            v-model="param.password"
            @keyup.enter="submitForm(login)"
          >
            <template #prepend>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <div class="login-btn">
          <el-button type="primary" @click="submitForm(login)">登录</el-button>
        </div>
        <p class="login-tips">Tips : {{ message }}</p>
        <el-alert type="warning">123456</el-alert>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Lock, User } from '@element-plus/icons-vue'

import { isDebug } from '../utils'
import { useTagsStore } from '../store/tags'
// eslint-disable-next-line camelcase
import { login_svc } from '../api/sys/user'
import { setToken, Storage, SessionKey, showLoading, closeLoading } from 'co6co'

import { registerRoute } from '../router/hooks'
import type { FormInstance, FormRules } from 'element-plus'
interface LoginInfo {
  username: string
  password: string
}
let message = ref('')
const param = reactive<LoginInfo>({
  username: '', // "admin",
  password: '' //"admin12345"
})
if (isDebug) {
  param.username = 'admin'
  param.password = 'admin12345'
}

const rules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}
let storeage = new Storage()
const login = ref<FormInstance>()
const submitForm = (formEl: FormInstance | undefined) => {
  if (!formEl) return
  formEl.validate((valid: boolean) => {
    if (valid) {
      showLoading()
      login_svc({ userName: param.username, password: param.password })
        .then((res) => {
          message.value = res.message
          if (res.code == 0) {
            setToken(res.data.token, res.data.expireSeconds)
            storeage.set('userName', param.username, res.data.expireSeconds)
            storeage.set(SessionKey, res.data.sessionId, res.data.expireSeconds)
            registerRoute(() => {
              window.location.href = '/audit'
              /*
								router.push('/'); 
								ElMessage.success(message.value);
								closeLoading();
								*/
            })
          } else {
            ElMessage.error(message.value)
          }
        })
        .catch((err) => {
          //todo debug  login.vue?t=1697628463669:82 Uncaught (in promise) TypeError: Cannot read properties of undefined (reading 'message')
          message.value = err.message || '请求出错'
          ElMessage.error(err.message)
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
  background-image: url(../assets/img/login-bg.jpg);
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
