<template>
  <el-button-group>
    <el-button text>
      <router-link to="/service">
        <el-icon style="color: red">
          <Histogram />
        </el-icon>
        <span>业务办理</span>
      </router-link>
    </el-button>
    <el-button text>
      <router-link to="/easyService">
        <el-icon style="color: red">
          <School />
        </el-icon>
        <span>便民服务</span>
      </router-link>
    </el-button>
    <el-button text>
      <router-link to="/userInfo">
        <el-icon>
          <User />
          <Avatar />
        </el-icon>
        <span>个人中心</span>
      </router-link>
    </el-button>
  </el-button-group>
</template>
<script setup>
import { ElButton, ElButtonGroup } from 'element-plus'
import { Histogram, School, User } from '@element-plus/icons-vue'
</script>
<style scoped lang="less">
.el-button-group {
  padding: 5px;
  display: flex;
  vertical-align: middle;
  height: 90%;
  .el-button {
    .el-icon {
      vertical-align: middle;
      margin-bottom: 5px;
      font-size: 30px;
      color: blue;
    }
    span {
      display: block;
      color: #010101;
    }
    border: 0;
    width: 33.3%;
    background: initial;
  }
}
</style>
