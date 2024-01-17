<template >
	<div class="ptz-controls show">
		<div class="ptz-bg-active"></div>
		<div
			class="ptz-arrow ptz-arrow-up"
			:class="{ disabled: !ptzEnable }"
			data-arrow="up"
			@mousedown="onOperater('up', 0)"
			@mouseup="onOperater('up', 1)"></div>
		<div
			class="ptz-arrow ptz-arrow-right"
			:class="{ disabled: !ptzEnable }"
			data-arrow="right"
			@mousedown="onOperater('right', 0)"
			@mouseup="onOperater('right', 1)"></div>
		<div
			class="ptz-arrow ptz-arrow-down"
			:class="{ disabled: !ptzEnable }"
			data-arrow="down"
			@mousedown="onOperater('down', 0)"
			@mouseup="onOperater('down', 1)"></div>
		<div
			class="ptz-arrow ptz-arrow-left"
			:class="{ disabled: !ptzEnable }"
			data-arrow="left"
			@mousedown="onOperater('left', 0)"
			@mouseup="onOperater('left', 1)"></div>
		<div
			class="ptz-control"
			@click="onCenter"
			:class="{ active: centerState, disabled: !takerEnable }">
			<el-icon><Microphone /> </el-icon>
		</div>
		<el-slider  v-model="speed" vertical placement="bottom" :min="1" height="156px"  :max="255" style="position: absolute; left:156px;top:0;   --el-slider-main-bg-color: #696970; " />
		<div class="ptz-btns">
			<div class="ptz-btn">
				<div
					class="ptz-expand ptz-icon"
					:class="{ disabled: !ptzEnable }"
					@mousedown="onOperater('zoomin', 0)"
					@mouseup="onOperater('zoomin', 1)">
					<i class="ptz-expand-icon"></i>
					<span class="icon-title-tips">
						<span class="icon-title">缩放+</span>
					</span>
				</div>
				<div
					class="ptz-narrow ptz-icon"
					:class="{ disabled: !ptzEnable }"
					@mousedown="onOperater('zoomout', 0)"
					@mouseup="onOperater('zoomout', 1)">
					<i class="ptz-narrow-icon"></i>
					<span class="icon-title-tips">
						<span class="icon-title">缩放-</span>
					</span>
				</div>
			</div>
		</div>
		
	</div>

	<div class="zoom-controls">
		<div class="zoom-narrow" :class="{ disabled: !ptzEnable }">
			<i class="icon icon-narrow"></i>
			<span class="icon-title-tips"><span class="icon-title">缩小</span></span>
		</div>
		<div class="zoom-tips">电子放大</div>
		<div class="zoom-expand" :class="{ disabled: !ptzEnable }">
			<i class="icon icon-expand"></i>
			<span class="icon-title-tips"><span class="icon-title">放大</span></span>
		</div>
		<div class="zoom-stop2">
			<i class="icon icon-zoomStop"></i>
			<span class="icon-title-tips"
				><span class="icon-title">关闭电子放大</span></span
			>
		</div>
	</div>
</template>
<script setup lang="ts">
	import * as t from './types/ptz';
	import { ref, watch, computed ,onMounted} from 'vue';
	import { connected, nextTick } from 'process';

	const props = defineProps({
		ptzEnable: {
			type: Boolean,
			default: true,
		},
		takerEnable: {
			type: Boolean,
			default: true,
		},
		takerState: {
			type: Boolean,
			default: true,
		},
	});
	interface Emits {
		(e: 'ptz', name: t.ptz_name, type: t.ptz_type,speed:number): void; 
		(e: 'centerClick', active: boolean): void;
	}
	const emits = defineEmits<Emits>();
	const starting=ref(false)
	const onOperater = (name: t.ptz_name, type: number) => {
		if (!props.ptzEnable) return; 
		if(type === 0){
			starting.value=true
			emits('ptz', name,  'starting'  ,speed.value); 
		} 
		else {
			starting.value=false 
			//给一个Tick 发送 stop
			nextTick(()=>{
				emits('ptz', name,  'stop',speed.value); 
			})
		}  
	};

	const centerActive = ref(false);
	const centerState = computed(() => {
		if (props.takerState && centerActive.value) {
			return true;
		}
		return false;
	});

	const onCenter = () => {
		if (!props.takerEnable) return;
		centerActive.value = !centerActive.value;
		emits('centerClick', centerActive.value);
	};

	const speed=ref(240) 
	onMounted(()=>{ 
		document.body.addEventListener("mouseup", ()=>{ 
			if(starting.value) {
				//拖动鼠标等才会出现
				console.warn("出现没有start的stop.")
				starting.value=false
				emits('ptz', 'up','stop',speed.value); 
			} 
		});
	})
</script>
<style scoped lang="less">
	@normal-color: #fff;
	@active-color: #706161;

	.inverted {
		filter: invert(100%);
	}
	.ptz-controls {
		position: relative;
		width: 156px;
		height: 156px;
		visibility: hidden;
		opacity: 0;
		border-radius: 78px;
		background-size: 100% 100%;
		transition: visibility 0.3s, opacity 0.3s;
		background: url('data:image/png;base64,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')
			no-repeat 50%;
	}
	.ptz-controls.show {
		visibility: visible;
		opacity: 1;
	}
	.ptz-bg-active {
		visibility: hidden;
		opacity: 0;
		width: 156px;
		height: 156px;
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAATgAAAE4CAMAAAD4oR9YAAAAM1BMVEX///////////////////////////9HcEz///////////////////////////////////85yRS0AAAAEXRSTlO5DCgVgZBxAK2fQDRkBR5XTPLKM/gAABnnSURBVHja7F0Jkqs6DGQPO9z/tD8Jq6WWbCCvIjKfzAGmulrW3ooedr6ui+M4TdP++SXPr1l/SdL3aRrHhv7ZyA5qb9xe0L3Am+DrkzeCL/BeX908MezTuPsfOArdgl3KsZuhq99fk/Tx3waum+ByAHua5QbYilkzY1aP728YhrH5InrfBa57OLAtVjpRbYaumex04dq4APeC7vnVSfo/45bXLe33jGscMx3f0A1vyg3t69e2dRL/NeA6wrgdcCvjyPM2U25mXDt9xVD3f/qN0yi3Mm6P20S54vlXtGPS/R3GPSbYOsC4ZAvmJtiaGiL3Zlzx/Ht+Y/KXTJXbqmaqe9za1VYn3N7YpX/OVGev2qduOLIiB7xqOzGuWCiXFVmWtU3368A5lkqeOJI21I5XXaORxVRnxmUTdNnY/4U3riNvHMJts9XRtdXVUttipdzrK/4x7UyY6sK4Gbo+nU21T1zKcd9AGJetlMvyLKvj3zXVfeqQElMljINx3MK4xVQ3xj2Ry7N/6CiMOIfYyVUXWxUyBx7HuZRbcHt9bf/Lb9zsHlzKzabauJaK47iVcC7jJujS33/joKkmxDnM4QiJ4xDjZuT+DXQW3jgxV012qcPuiePhCGfchlv1/P0D6Czmqmuq2gPGkbIS8Q4ZsNU3dGP3Y2+cW1RyKpkrbAnwqhi3iXHFHrU3bFVV5c3vBsCsOALjkXErAW85F3rjFvBm5Kos+TngCOXYG7fA1ojFER7GPUHbmer0tfGPANeROM6pjvDMQSkrsWQ1d564Fbr61964TvSqDa6O0ELmAtvGuc2rrpQrn/aa/qCpYq+6mSpOVhnjWBy38u2JXFl9yL8acg6CV3Ur5yxVZfW4AsRxG+XKssz6n3njVMYR4Eg8sj1yi3tgtroxrpyhG38gc+h8PYddPQ551dVQW5jju2/cG7kXdB946Uy9cbDnQOpxuCCHcq5dHDcht8D2/K67VxPNGtLJd7qDTcgb1zLGbXEcY9z0Fd39GReTzIH1B/2lcxrGTYxj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')
			no-repeat 50%;
		background-size: 100% 100%;
	}
	.ptz-bg-active.ptz-bg-active-up {
		transform: rotate(-90deg);
	}
	.ptz-bg-active.ptz-bg-active-left {
		transform: rotate(180deg);
	}
	.ptz-bg-active.ptz-bg-active-down {
		transform: rotate(90deg);
	}
	.ptz-bg-active.ptz-bg-active-left-up {
		transform: rotate(-135deg);
	}
	.ptz-bg-active.ptz-bg-active-right-up {
		transform: rotate(-45deg);
	}
	.ptz-bg-active.ptz-bg-active-left-down {
		transform: rotate(135deg);
	}
	.ptz-bg-active.ptz-bg-active-right-down {
		transform: rotate(45deg);
	}
	.ptz-bg-active.ptz-bg-active-show {
		visibility: visible;
		opacity: 1;
	}

	@keyframes anim1 {
		25% {
			color: darken(red, 20%);
		}
		50% {
			color: darken(red, 10%);
		}
	}
	.ptz-control {
		cursor: pointer;
		position: absolute;
		left: 53px;
		top: 53px;
		width: 50px;
		height: 50px;
		background: @normal-color;
		border-radius: 50%;
		transition: left 0.3s, top 0.3s;
		&.active {
			background: @active-color;
			.el-icon {
				color: darken(@normal-color, 10%);
				animation-name: anim1;
				animation-direction: alternate;
				animation-timing-function: linear;
				animation-delay: 0s;
				animation-iteration-count: infinite;
				animation-duration: 2s;
			}
		}
		&.disabled {
			background: @normal-color;
		}
		//filter: invert(52%) sepia(82%) saturate(2494%) hue-rotate(327deg) brightness(104%) contrast(92%);
		.el-icon {
			position: absolute;
			left: 18px;
			top: 18px;
		}
	}
	.ptz-control.ptz-control-left {
		left: 33px;
	}
	.ptz-control.ptz-control-up {
		top: 33px;
	}
	.ptz-control.ptz-control-right {
		left: 73px;
	}
	.ptz-control.ptz-control-down {
		top: 73px;
	}
	.ptz-control.ptz-control-left-up {
		top: 39px;
		left: 39px;
	}
	.ptz-control.ptz-control-left-down {
		left: 39px;
		top: 67px;
	}
	.ptz-control.ptz-control-right-up {
		top: 39px;
		left: 67px;
	}
	.ptz-control.ptz-control-right-down {
		top: 67px;
		left: 67px;
	}

	.ptz-icon {
		position: relative;
	}
	.ptz-icon:hover .icon-title-tips {
		visibility: visible;
		opacity: 1;
	}
	.ptz-btns {
		display: block;
		position: absolute;
		left: 0;
		top: 156px;
		width: 156px;
		box-sizing: border-box;
		padding: 0 30px;
	}
	.ptz-btns .ptz-btn {
		display: flex;
		justify-content: space-between;
	}
	.ptz-expand .ptz-expand-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		cursor: pointer;
		/** + */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAM1BMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn////i4uLZ2dnIyMjExMS8vLy+iXNeAAAACnRSTlMAYomLxwEm9+NCLo6DKwAAALNJREFUKM99k9kWgyAMRIMmEMLm/39tKaVKFJkXl3sYJ4sAXeQ3ZOcYd0+gRYblFBuFLYoS2ot5lpvYn8zJQ65TO2GVNmdCmQq/qczw4gjpejD14BgmhziEIvCjVRlPioftHW6A7xBB1a8CCUMvsuSqEkPM7eZX6h8GrQ67bYpNIbRL6rb4/k2EfVXKsgmqfQrW9qnGq96a28jGQG1ky2HXpVysyYyeDIhWq7le6ua9P36HD6+2GRi8iBZBAAAAAElFTkSuQmCC')
			no-repeat 50%;

		background-size: 100% 100%;
	}
	.ptz-expand:hover .ptz-expand-icon {
		/** + */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAANlBMVEVfX19fX19fX19fX19fX19fX19fX19HcExfX19fX19fX1/////Pz8+oqKjCwsKhoaHn5+eWlpaOqTaDAAAAC3RSTlP/3CaKxwJiAELtp4ri/s4AAACuSURBVCjPfZPREoUgCERXBdPRyv7/Z6/Z1aQp9oWJMyYLiKUrOIpAJBdGCldgbzBkPM/QEoTI3jBEPBRDhwEvChe08Q1Ge0ImvIq4Qj8ljrLdH77CyQPWlCdHC0Q1e9rmmuC+oQN9Q4LwcQg40L6eyqm0uEpXSUqe3fKpkkqL+Y/o+07SrahNEO0T0LBsvOitf4xsLqiNTB32wtqaVKosGLO2mhUrS93+PZ4D99wPqzMJVcbEyA8AAAAASUVORK5CYII=')
			no-repeat 50%;
	}
	.ptz-expand.disabled .ptz-expand-icon {
		/** + */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAM1BMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn////i4uLZ2dnIyMjExMS8vLy+iXNeAAAACnRSTlMAYomLxwEm9+NCLo6DKwAAALNJREFUKM99k9kWgyAMRIMmEMLm/39tKaVKFJkXl3sYJ4sAXeQ3ZOcYd0+gRYblFBuFLYoS2ot5lpvYn8zJQ65TO2GVNmdCmQq/qczw4gjpejD14BgmhziEIvCjVRlPioftHW6A7xBB1a8CCUMvsuSqEkPM7eZX6h8GrQ67bYpNIbRL6rb4/k2EfVXKsgmqfQrW9qnGq96a28jGQG1ky2HXpVysyYyeDIhWq7le6ua9P36HD6+2GRi8iBZBAAAAAElFTkSuQmCC')
			no-repeat 50%;
		cursor: not-allowed;
	}
	.ptz-narrow .ptz-narrow-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		/* - */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcBAMAAACAI8KnAAAAJ1BMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn+/v7X19ckk9ihAAAACnRSTlMA9+NCAsuKJsRiPv/2GwAAAJlJREFUGNNjYAAC5gxFoTYDBijw1FoFBIumQHjsUavAYGkBmGu0CgqUwRqlYNyFIO2Fq+BAnIGBJQrBXerAwLkKCUxgYELmKjBYIXMXM2Qhc5cxdCFzVzBoIXMXMYAcsRsMdgEdgs4FKT4DBqdAitGMQrMIzRkojlRB9wKaB9G8z+CMGjgshjCuMCjoWNxRAxYt2KGRYgJiAQAnZcjElaB/xwAAAABJRU5ErkJggg==')
			no-repeat 50%;

		background-size: 100% 100%;
		cursor: pointer;
	}
	.ptz-narrow:hover .ptz-narrow-icon {
		/* - */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAM1BMVEVHcExfX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX1/9/f2/v7/y8vLUObqxAAAADXRSTlMA3IrE6SZi9wI+y0gNXAn3CgAAAI5JREFUKM+Fk1kOwyAMBQ04bJHT3P+0JVUMNMWv8zvSk1cipfjAKXHwhR7k6KTjYp7dVuWLug1XWB5wz96T/JD2O3Phmv0k5ypL6lVVFIPYpLOka5WKSSFvS0/BloHYlkza5HkMzrvVLo8ZlRr7mtFYWBBsBQ4BjC//GTxcGVw2PpOVHQ6fJj7qS4936OoN2K4e5yE6N1UAAAAASUVORK5CYII=')
			no-repeat 50%;
	}
	.ptz-narrow.disabled .ptz-narrow-icon {
		/**- */
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcBAMAAACAI8KnAAAAJ1BMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn+/v7X19ckk9ihAAAACnRSTlMA9+NCAsuKJsRiPv/2GwAAAJlJREFUGNNjYAAC5gxFoTYDBijw1FoFBIumQHjsUavAYGkBmGu0CgqUwRqlYNyFIO2Fq+BAnIGBJQrBXerAwLkKCUxgYELmKjBYIXMXM2Qhc5cxdCFzVzBoIXMXMYAcsRsMdgEdgs4FKT4DBqdAitGMQrMIzRkojlRB9wKaB9G8z+CMGjgshjCuMCjoWNxRAxYt2KGRYgJiAQAnZcjElaB/xwAAAABJRU5ErkJggg==')
			no-repeat 50%;
		cursor: not-allowed;
	}
	.ptz-aperture-far .ptz-aperture-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAASFBMVEVHcExfX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX1////9fX1+kpKRzc3ODg4PFxcW1tbXW1tbk5OT29vaVlZVmZmZ8vCMFAAAADHRSTlMAxGJ5Mssm9+NCiYtiH91SAAABAklEQVQoz41T7Q6DIAyEJQooUL55/zddhVazzZjdHyqXXo8DhWCYTWqltNyN+MZLuxP69UGti/vAsl6c0e4L+tQ2yv1AEbvecMhO5cXdYhk+6aO3WGrNAMwentlMz/ZAKIlNoRsqY2wtFWu9t8wasc0iYVN0LkQfrG1zbxNyrIBcntOQrH1Ukkb60QcxYF1xMA2dh8zWj6ZDsLCsIrL4Ds5Hm9FMbCEROWUB0COaLXEIZJKV7CKybGO7UuxjxY2C/TkMbxboKBQCxgMN6MCJQ6Ch/QjOZg/B13LGx8FDTe3IFvl+Bc9XBi3UWoex68qeL/vxmdyxyvz3NJ8f9dDef36HN7koIK2LjxB0AAAAAElFTkSuQmCC')
			no-repeat 50%;
		background-size: 100% 100%;
		cursor: pointer;
	}
	.ptz-aperture-far:hover .ptz-aperture-far.active .ptz-aperture-icon {
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAOVBMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn+/v6cnJzr6+u/v7+xsbGlpaXNzc3b29vqh7uRAAAAC3RSTlMAyRjKA59J/3PzPhe1wxwAAAD2SURBVCjPjZPrssMgCIR1mkYtisD7P+zBCyZpM5mzv5hs0M8VnZvaok9BlXzc3FXbO5z0vtifFC5Kn8OL4UfxwVvuHm61d5Z0b6ZGZZwZpQAUosWsjVZntVS1sH3ZFo1IRVYfGXgx+VGwNkkIVbhq9/jm3cAhaNv1Uk3IA8mNn7D3kbQeWK3TLH2jCthrDFcTMwUWaKiClc9mJtJWhS3SF5BpJqMQW1b3xwnkDahMoHYomkeJRgSENA/MFsKML7fgoCBVbGvM+Cx4JcKWbWHKK/h1ZYS1Jy/nK3u8bB3KhzG5deMxtfv3aO7/Heq+9ms8h9fxHP4AHzAWU9zlWNgAAAAASUVORK5CYII=')
			no-repeat 50%;
		background-size: 100% 100%;
	}
	.ptz-aperture-near .ptz-aperture-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAQlBMVEVHcExfX19fX19fX19fX19fX19fX19fX19fX19fX19fX1////9fX1+FhYWbm5vz8/Nzc3OwsLDi4uLDw8PW1tZmZmYgm6a+AAAAC3RSTlMAYmOLx4kn9+NCIVJiPGAAAAD+SURBVCjPjVPttoMgDMOJAqOU8uH7v+qKFN2c597lF5LTJg1VqQG3aGuM1bNTV0wWDtjpg3pq+IB+npyzcIE9ejsDXzDCrjccs+tOariF3n2OLyw5xko0vh9MDjNb9Q0hp2GK3cixlIApe4/JD9appR8SFxWAUFLg6n63iB1irnY1Jv0mlrok7nUdcZRa1YeshxBA9iijChlxI6iZEaBgSEL2tkRcymPGGJpqlbZ6uDg0WR/F0DwuMpxDkYwiIXA8hO2uMJdGCCK6teB8RQoY8xGfevQjxYQt25qoRwDT25MRBjZ7GtP/P/afa3LHmrflXa+ruf661Hvv+et3eAF6Fh3v+sSUGgAAAABJRU5ErkJggg==')
			no-repeat 50%;
		background-size: 100% 100%;
		cursor: pointer;
	}
	.ptz-aperture-near:hover .ptz-aperture-near.active .ptz-aperture-icon {
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAM1BMVEWZmZmZmZmZmZmZmZmZmZmZmZmZmZlHcEyZmZmZmZn///+qqqq9vb3z8/PMzMzo6Oja2tpXGg+mAAAACnRSTlP/JomLxwJiAONCr+rW2wAAAOtJREFUKM99U9sWhCAInEpLBS///7WLEWy7p9O8qEzCMBIOQ15DAlLYsoegS9yFMKQ93skl4Adh+ZI54Q8pG5nxgKzkgkcsk4zhmQxRyN1OPHqtncjOu5AuppcJ6s1EHTA1YzC3Wgq3YmzGqpsmlwZAo7F8oLEVKoeE6+TbSxK0JJ/3FLOwFnUxzXuoltYDDMLoAlmYXLAWIrkqbdZKs+q4KBfkNV1uwGaBim9TdLWS3R7iGRvCNTPB7JvGlc5EXK8cKbrxooint73RzXh7Msl6Oj/uT/b62O9j8sj6gMXX0Xwf6jP3Zr9DtNAHTYMMXrXSK0YAAAAASUVORK5CYII=')
			no-repeat 50%;
		background-size: 100% 100%;
	}
	.ptz-focus-far .ptz-focus-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAPFBMVEVHcExfX19fX19fX19fX19fX19fX19fX19fX19fX19fX1////92dnbs7OyFhYWjo6Pe3t7Ly8uxsbG8vLyG+Q0EAAAAC3RSTlMAiWJjx9wm/0Lti7mfpe0AAADaSURBVCjPfVMJDoQgDERFC/bg8P9/XUSO6CqTqA0TptNDpSrMpC2A1btRT8wrNKzzjdo03KC3zhkLD9imbeAFhd3sG2kvZQ2v0NknfGBJZKkhBM9MxOxDKBV1N4iHi0TRHYjN01Qi7/kK2PtyNDU7DAEJgDAAN0u1jsQEFEkcVVmrqjeXrkWRmC67eqbgG7bJyvkQSQkvUvec7szpek6t9ubWJSK/uJVSm+APzHKCh++DWWuH4plQKNYOpfappcjy2VvJn9744cjGwx6uyXjBxqs5Xuqsvf/9Dj8rLhRg+bQ5VAAAAABJRU5ErkJggg==')
			no-repeat 50%;
		background-size: 100% 100%;
		cursor: pointer;
	}
	.ptz-focus-far:hover .ptz-focus-far.active .ptz-focus-icon {
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAOVBMVEWZmZmZmZmZmZmZmZmZmZmZmZmZmZlHcEyZmZmZmZmZmZn///+xsbGoqKjt7e309PTExMTQ0NDe3t774OlGAAAAC3RSTlP/itxixwImAELtp8B2gZgAAADmSURBVCjPjZMLjsUgCEVpq60G8Lf/xQ62gvNeOmZuUiWeKHC1cKnC5iJAdFuwJXgmf+xg2g//G54OPuTOCUOEL8WgMMCLwgPP+Abj2aF38CrnBR7whw6Bo4fWUk7MMrQ2OrpAq0GspTLLgKg1wTailNITZA0EaTkZGjIAY5NwlATah5CGRMJYj50tFtlWiapsLvAPRdtL/WOmET7QzZyl5ywzp7NWsjBJ1odsragJqeJ9HGFNZoLaJw71hMTm0O7NeDE1Z6YsU5rGL69sedmXXz0ToW8PzA/oV09T8OJR32fb7+B17Qe3WwtC9PVbHAAAAABJRU5ErkJggg==')
			no-repeat 50%;
		background-size: 100% 100%;
	}
	.ptz-focus-near .ptz-focus-icon {
		display: inline-block;
		width: 28px;
		height: 28px;
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAARVBMVEVHcExfX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX19fX1/////t7e2dnZ3W1tbGxsa3t7eDg4Oqqqri4uKTk5NImu/5AAAADXRSTlMAYieJ3MvE/0Lti4oh87zNagAAAOtJREFUKM+NU1sOwyAMY30FtoWS8Lj/UZe2gWpVh2aJH1wcO0mNqbDj4gDc8rLmiscEDdPji3rP8IX5fXLWwQWuaVu4gbKDuyPdsJMz3GLefcIPbJ6PDCEAFDlUAJiORM3NigQFAXAFlqOeRhWJyFFIHxNGvRrN0mp470U++3axGM2RAmXcXqKnkDSN0a9WIk5Sa01MpDXBQAdVtrA8lBhFnnKpsmoo5VBrhszV0KuJ5N2tP92O50iQjpzcctravoihdoi0Q1NrfN56m0VWzFBoje+OrD/s7pr0F0yUr6s5/LvUu/bz+B2ep+IHdMIV2SUZfCsAAAAASUVORK5CYII=')
			no-repeat 50%;
		background-size: 100% 100%;
		cursor: pointer;
	}
	.ptz-focus-near:hover .ptz-focus-near.active .ptz-focus-icon {
		background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAQlBMVEVHcEyZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZn////c3Nz09PTp6enR0dHFxcW7u7uwsLAUKT0cAAAADXRSTlMA3IrE6WIm9wI+y0gNQZpqdwAAAOdJREFUKM99U9GShCAMQ8BF3Cmlpfz/r15dAe88l8zwQiQkoRrTEa3zIXhno7lhWxcYWNbtN/fa4Q/218VFDzf4of0O8A/h3TQfOGU/ytsOj9gPVyt8warkmYEQQAgABYDxTKROz88koS6AVIB1fRCNbSI1cVUy15Jq27LGjTtyzipPeWw40/IXQkrHyZSRmqw3LaQgctFNKYzYyGACfEXossLMojFEj7J0WfdwJ3dD9uY2X25tL0Hj45mTR87Y66u9IQFsDS1bL57o7JbUDNIofvpk08eej8kTe3Hz0ZwP9UFfv8OgfgBUByCEUZhYtAAAAABJRU5ErkJggg==')
			no-repeat 50%;
		background-size: 100% 100%;
	}
	.ptz-arrow {
		cursor: pointer;
		position: absolute;
		width: 0;
		height: 0;
	}
	.ptz-arrow-up {
		left: 71px;
		top: 15px;
		border: 7px solid transparent;
		border-bottom: 10px solid @normal-color;
		&:hover,
		&.active {
			border-bottom: 10px solid @active-color;
		}
		&.disabled {
			border-bottom: 10px solid @normal-color;
		}
	}

	.ptz-arrow-right {
		top: 71px;
		right: 15px;
		border: 7px solid transparent;
		border-left: 10px solid @normal-color;
		&:hover,
		&.active {
			border-left: 10px solid @active-color;
		}
		&.disabled {
			border-left: 10px solid @normal-color;
		}
	}
	.ptz-arrow-left {
		left: 15px;
		top: 71px;
		border: 7px solid transparent;
		border-right: 10px solid @normal-color;
		&:hover,
		&.active {
			border-right: 10px solid @active-color;
		}
		&.disabled {
			border-right: 10px solid @normal-color;
		}
	}
	.ptz-arrow-down {
		left: 71px;
		bottom: 15px;
		border: 7px solid transparent;
		border-top: 10px solid @normal-color;
		&:hover,
		&.active {
			border-top: 10px solid @active-color;
		}
		&.disabled {
			border-top: 10px solid @normal-color;
		}
	}
	.ptz-arrow-left-up {
		transform: rotate(45deg);
		left: 32px;
		top: 33px;
		border: 7px solid transparent;
		border-right: 10px solid @normal-color;
		&:hover,
		&.active {
			border-right: 10px solid @active-color;
		}
		&.disabled {
			border-right: 10px solid @normal-color;
		}
	}
	.ptz-arrow-right-up {
		transform: rotate(-45deg);
		right: 32px;
		top: 33px;
		border: 7px solid transparent;
		border-left: 10px solid @normal-color;
		&:hover,
		&.active {
			border-left: 10px solid @active-color;
		}
		&.disabled {
			border-left: 10px solid @normal-color;
		}
	}
	.ptz-arrow-left-down {
		transform: rotate(45deg);
		left: 32px;
		bottom: 33px;
		border: 7px solid transparent;
		border-top: 10px solid @normal-color;
		&:hover,
		&.active {
			border-top: 10px solid @active-color;
		}
		&.disabled {
			border-top: 10px solid @normal-color;
		}
	}
	.ptz-arrow-right-down {
		transform: rotate(-45deg);
		right: 32px;
		bottom: 33px;
		border: 7px solid transparent;
		border-top: 10px solid @normal-color;
		&:hover,
		&.active {
			border-top: 10px solid @active-color;
		}
		&.disabled {
			border-top: 10px solid @normal-color;
		}
	}
	.zoom-controls {
		display: none;
		position: absolute;
		left: 50%;
		top: 0;
		padding: 0 3px;
		transform: translateX(-50%);
		justify-content: space-around;
		align-items: center;
		width: 150px;
		height: 30px;
		background: #000;
		opacity: 1;
		border-radius: 0 0 8px 8px;
		z-index: 1;
	}
	.zoom-controls .zoom-narrow {
		width: 16px;
		height: 16px;
		cursor: pointer;
	}
	.zoom-controls .zoom-tips {
		font-size: 14px;
		font-weight: 500;
		color: #ddd;
	}
	.zoom-controls .zoom-expand,
	.zoom-controls .zoom-stop2 {
		width: 16px;
		height: 16px;
		cursor: pointer;
	}
	.loading {
		display: none;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		position: absolute;
		z-index: 20;
		left: 0;
		top: 0;
		right: 0;
		bottom: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
	.loading-text {
		line-height: 20px;
		font-size: 13px;
		color: #fff;
		margin-top: 10px;
	}
	.controls {
		background-color: #161616;
		box-sizing: border-box;
		display: flex;
		flex-direction: column;
		justify-content: flex-end;
		position: absolute;
		z-index: 40;
		left: 0;
		right: 0;
		bottom: 0;
		height: 38px;
		width: 100%;
		padding-left: 13px;
		padding-right: 13px;
		font-size: 14px;
		color: #fff;
		opacity: 0;
		visibility: hidden;
		transition: all 0.2s ease-in-out;
		-webkit-user-select: none;
		-moz-user-select: none;
		user-select: none;
		transition: width 0.5s ease-in;
	}
	.controls .controls-item {
		position: relative;
		display: flex;
		justify-content: center;
		padding: 0 8px;
	}
	.controls .controls-item:hover .icon-title-tips {
		visibility: visible;
		opacity: 1;
	}

	.controls .controls-item.face,
	.controls .controls-item.face-active,
	.controls .controls-item.fullscreen,
	.controls .controls-item.fullscreen-exit,
	.controls .controls-item.icon-audio,
	.controls .controls-item.microphone-close,
	.controls .controls-item.pause,
	.controls .controls-item.performance,
	.controls .controls-item.performance-active,
	.controls .controls-item.play,
	.controls .controls-item.ptz,
	.controls .controls-item.ptz-active,
	.controls .controls-item.quality-menu,
	.controls .controls-item.record,
	.controls .controls-item.record-stop,
	.controls .controls-item.scale-menu,
	.controls .controls-item.screenshot,
	.controls .controls-item.speed-menu,
	.controls .controls-item.template-menu,
	.controls .controls-item.volume,
	.controls .controls-item.zoom,
	.controls .controls-item.zoom-stop {
		display: none;
	}
	.controls .controls-item-html {
		position: relative;
		display: none;
		justify-content: center;
	}
	.controls .icon-audio,
	.controls .icon-mute {
		z-index: 1;
	}
	.controls .controls-bottom {
		display: flex;
		justify-content: space-between;
		height: 100%;
	}
	.controls .controls-bottom .controls-left,
	.controls .controls-bottom .controls-right {
		display: flex;
		align-items: center;
	}
	.controls-show .controls {
		opacity: 1;
		visibility: visible;
	}
	.controls-show-auto-hide .controls {
		opacity: 0.8;
		visibility: visible;
		display: none;
	}
	.icon-title-tips {
		pointer-events: none;
		position: absolute;
		left: 50%;
		bottom: 100%;
		visibility: hidden;
		opacity: 0;
		transform: translateX(-50%);
		transition: visibility 0.3s ease 0s, opacity 0.3s ease 0s;
		background-color: rgba(0, 0, 0, 0.5);
		border-radius: 4px;
	}
	.icon-title {
		display: inline-block;
		padding: 5px 10px;
		font-size: 12px;
		white-space: nowrap;
		color: #fff;
	}

	.disabled {
		cursor: not-allowed;
	}
</style>
