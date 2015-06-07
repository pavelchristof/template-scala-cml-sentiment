from django.views.generic import TemplateView
from django.shortcuts import render
from predictionio import EngineClient


def add_colors(node):
    r = dict()

    if "value" in node:
        r["label"] = node["value"]
    else:
        r["children"] = map(add_colors, [node["left"], node["right"]])

    vec = node["accum"]["get"]
    s = max([(t[1], t[0]) for t in enumerate(vec)])[1]
    colors = ["red", "orange", "white", "LightGreen", "green"]
    r["color"] = colors[s]

    return r


class IndexView(TemplateView):
    template_name = "sentiment/index.html"

    def post(self, request):
        engine = EngineClient("http://localhost:8001")
        result = engine.send_query({'sentence': request.POST.get('sentence', '')})
        return render(request, 'sentiment/prediction.html', {'result': result, 'node': add_colors(result["sentence"])})