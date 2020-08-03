try {
    require.undef("validate_widget")

    define('validate_widget', ["@jupyter-widgets/base"], function(widgets) {
        var ValidateView = widgets.DOMWidgetView.extend({
            render () {
                window.widget_self = this
                var visualize_id = this.model.get('visualize_id')

                if (!window._renderLock) {
                    window._renderLock = {}
                }
                if (window._renderLock[visualize_id]) {
                    return
                }
                window._renderLock[visualize_id] = "widget"
                console.log("load as widget", Date.now())

                var lib_url = this.model.get('lib_url')
                var graph_json = JSON.parse(this.model.get('graph_json'))
                var env_json = JSON.parse(this.model.get('env_json'))
                var container_id = this.model.get('container_id')

                window.render_container_id = container_id
                window.graph_json = graph_json
                window.env_json = env_json
                window.before_script = performance.now()

                var container = document.createElement('div')
                container.id = container_id
                this.el.appendChild(container)

                var style = document.createElement('style')
                style.innerHTML = [
                    "#", container_id, " svg.react-dag-editor-svg-container { height: 800px; }",
                    ".cell-output-ipywidget-background { background: transparent !important }"
                ].join('')
                this.el.appendChild(style)

                this.model.on('msg:custom', dispatchMessage, this);

                if (!window.__event_hub) {
                    window.__event_hub = {}
                }
                if (!window.__event_hub[container_id]) {
                    window.__event_hub[container_id] = {}
                }

                if (!window.__send_event) {
                    window.__send_event = {}
                }
                window.__send_event[container_id] = sendMessage.bind(this)

                function sendMessage(message, uid, content) {
                    return new Promise((resolve) => {
                        this.model.send({
                            message: `${message}:request`,
                            body: {
                                uid,
                                content
                            }
                        })
    
                        var respMessageKey = `${message}:response`
                        if (!window.__event_hub[container_id][respMessageKey]) {
                            window.__event_hub[container_id][respMessageKey] = []
                        }
                        window.__event_hub[container_id][respMessageKey].push(callback)
    
                        function callback (response) {
                            if (response.uid !== uid) {
                                return
                            }

                            var idx = window.__event_hub[container_id][respMessageKey].indexOf(callback) 
                            window.__event_hub[container_id][respMessageKey].splice(idx, 1)
                            
                            resolve(response)
                        }
                    })
                }

                function dispatchMessage (rawMessage) {
                    var message = rawMessage.message
                    var body = rawMessage.body

                    if (!window.__event_hub[container_id][message]) {
                        window.__event_hub[container_id][message] = []
                    }
                    var listeners = window.__event_hub[container_id][message]

                    listeners.forEach(cb => {
                        try {
                            cb(body)
                        } catch (e) {
                            console.error("Unexpected error in listener", e)
                        }
                    })

                    console.log(body)
                }

                var script = document.createElement('script')
                script.src = lib_url
                this.el.appendChild(script)
            }
        });

        return {
            ValidateView
        }
    })
} catch (e) {
    console.log("create validation widget failed", e)
}