<template>
	<div v-loading="loading">
		<div>
			<!--<v-tags></v-tags>-->
			<div class="content">
				<router-view v-slot="{ Component }">
					<transition name="move" mode="out-in">
						<!--<keep-alive :include="tags.nameList">-->
							<component :is="Component"></component>
						<!--</keep-alive>-->
					</transition>
				</router-view>
			</div>
		</div>
	</div>
</template>
<script setup lang="ts">
	import { ref, reactive, onMounted, WatchEffect, watchEffect } from 'vue'; 
    /*
    import { useTagsStore } from '../store/tags';
	import vTags from '../components/tags.vue';
	import { getToken, setToken } from '../utils/auth';
	import { ticket_svc } from '../api/authen';
	import { showNotify } from 'vant';
	import { getQueryVariable } from '../utils';
	import { redirectUrl } from '../components/wx';

	const tags = useTagsStore();
	let token = getToken();
	const tokenRef = ref(token);
	const ticket = getQueryVariable('ticket');
	if (ticket) showNotify({ type: 'success', message: ticket });
	const tokenChange = () => {
		if (tokenRef.value == null || ticket == null) {
			redirectUrl();
		}
	}; 
	watchEffect(() => {
		tokenChange();
	});

	if (ticket) {
		ticket_svc(ticket).then((res) => {
			if (res.code == 0)
				setToken(res.data),
					(tokenRef.value = getToken()),
					(loading.value = false);
			else showNotify({ type: 'danger', message: res.message });
		});
	}
    */
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
	.content {
		overflow: auto;
	}
	#app {
		overflow: auto;
	}
</style>
