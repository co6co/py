<template> 
	<div class="content" v-loading="loading">
		<router-view v-slot="{ Component }">
			<transition name="move" mode="out-in">
				 
				<component :is="Component"></component> 
			</transition>
		</router-view>
	</div>
</template>
<script setup lang="ts">
	import { ref, reactive, onMounted,   watchEffect } from 'vue'; 
	const loading = ref(true);
	onMounted(() => {
		loading.value = false;
	});
</script>
<style lang="less">
	body {
		margin: 0;
	}
	.example-showcase .el-loading-mask {
		z-index: 9;
	}

	#app {
		overflow: auto;
		.content {
			overflow: auto; 
			padding: 0; margin: 0;
			.el-container {
				height: 97vh;
				overflow: hidden;
			}
		}
	}
</style>
