<template>
    <el-card class="box-card">
        <template #header>
            <div class="card-header">
                <span>{{title}}</span> 
            </div>
        </template>
        <div style="height: 80%; overflow: auto;">
            <pre>
                {{ content }}  
            </pre>
        </div>
    </el-card>
</template>
<script setup lang="ts">
    const props = defineProps({ 
    title:{
        type: String,
        required: true
    },
    content: {
        type: String,
        required: true
    }}); 
</script>
<style scoped lang="less"> 
.el-card__body{height: 90%;}
</style>