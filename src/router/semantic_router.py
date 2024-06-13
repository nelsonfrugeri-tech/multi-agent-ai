from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route, RouteLayer



class SemanticRouter:
    def __init__(self):
        self.encoder = OpenAIEncoder("text-embedding-ada-002")

    def handler(self, context: str):
        route_layer = self._choice_agent()
        route = route_layer(context)
        if route:
            if route.name == "professor":
                return "call_gpt4"
            elif route.name == "greetings":
                return "call_llama3"
            elif route.name == "summarization_correction_feedback":
                return "call_gpt35_turbo"
        else:
            return "No suitable route found for the given prompt."

    def _choice_agent(self):
        return RouteLayer(
            encoder=self.encoder,
            routes=[
                Route(
                    name="greetings",
                    utterances=[
                        "Oi, tudo bem?",
                        "Olá! Posso ajudar em algo hoje?",
                        "Boa tarde! Em que posso ser útil?",
                        "Bom dia! Como posso ajudá-lo hoje?",
                        "Boa noite! Estou aqui para o que precisar.",
                        "Olá! Espero que esteja tudo ótimo com você!",
                        "Oi, como posso te ajudar hoje?",
                        "E aí,tudo certo?",
                        "Olá! Algum plano especial para hoje?",
                        "Bom dia! Alguma dúvida ou algo em que eu possa ajudar?",
                        "Boa tarde, está precisando de ajuda em algo?",
                        "Boa noite, há algo que eu possa fazer por você agora?",
                        "Oi, como está indo o seu dia?",
                        "Olá! Tudo tranquilo por aí?",
                        "Boa tarde! Precisa de alguma informação específica?",
                        "Bom dia! Estou à disposição para te ajudar.",
                        "Boa noite! Estou aqui para ajudar no que for necessário.",
                        "Oi! Como posso tornar seu dia melhor?",
                        "Olá, em que posso ser útil agora?",
                        "Oi! Se precisar de algo, é só chamar."
                    ]
                ),
                Route(
                    name="summarization_correction_feedback",
                    utterances=[
                        "Você pode resumir este texto para mim?",
                        "Como posso melhorar este parágrafo?",
                        "Quais são os pontos fortes deste texto?",
                        "Quais são os pontos fracos deste texto?",
                        "Você pode corrigir os erros gramaticais neste texto?",
                        "Como posso tornar este argumento mais convincente?",
                        "Você pode fornecer um feedback detalhado sobre esta redação?",
                        "Quais são as áreas que preciso melhorar nesta redação?",
                        "Você pode verificar a coesão deste texto?",
                        "Como posso melhorar a coerência deste texto?",
                        "Você pode sugerir melhorias para a conclusão deste texto?",
                        "Quais são os erros mais comuns que cometi neste texto?",
                        "Você pode revisar a estrutura deste texto?",
                        "Como posso tornar este texto mais claro e objetivo?",
                        "Você pode verificar se há repetição de ideias neste texto?",
                        "Quais são as sugestões para melhorar a introdução deste texto?",
                        "Você pode fornecer um resumo deste artigo?",
                        "Como posso melhorar a fluidez deste texto?",
                        "Você pode corrigir a pontuação neste texto?",
                        "Quais são as sugestões para melhorar a argumentação deste texto?",
                        "Você pode verificar se usei os conectivos corretamente?",
                        "Como posso melhorar a proposta de intervenção deste texto?",
                        "Você pode fornecer um feedback sobre a originalidade deste texto?",
                        "Quais são as sugestões para evitar clichês neste texto?",
                        "Você pode verificar se há erros de concordância neste texto?",
                        "Como posso melhorar a transição entre os parágrafos deste texto?",
                        "Você pode fornecer um resumo deste capítulo?",
                        "Quais são as sugestões para melhorar a conclusão deste texto?",
                        "Você pode verificar se há erros de ortografia neste texto?",
                        "Como posso melhorar a clareza dos meus argumentos neste texto?"
                    ]
                ),
                Route(
                    name="professor",
                    utterances=[
                        "O que é uma tese em uma redação do ENEM?",
                        "Como estruturar uma introdução eficaz?",
                        "Quais são os principais conectivos para usar na redação?",
                        "Como desenvolver um bom argumento?",
                        "O que é uma proposta de intervenção?",
                        "Como evitar a fuga ao tema?",
                        "Quais são os erros mais comuns na redação do ENEM?",
                        "Como fazer uma conclusão impactante?",
                        "O que é coesão textual?",
                        "Como usar citações na redação?",
                        "Quais são os critérios de correção da redação do ENEM?",
                        "Como melhorar a coerência do texto?",
                        "O que é um repertório sociocultural?",
                        "Como fazer uma redação nota 1000?",
                        "Quais são os tipos de argumentos que posso usar?",
                        "Como evitar a repetição de palavras?",
                        "O que é um texto dissertativo-argumentativo?",
                        "Como fazer a transição entre parágrafos?",
                        "Quais são os temas mais recorrentes na redação do ENEM?",
                        "Como usar dados estatísticos na redação?",
                        "O que é um argumento de autoridade?",
                        "Como fazer uma boa revisão do texto?",
                        "Quais são as competências avaliadas na redação do ENEM?",
                        "Como evitar o uso de clichês na redação?",
                        "O que é um parágrafo de desenvolvimento?",
                        "Como fazer uma introdução criativa?",
                        "Quais são as estratégias para argumentar de forma eficaz?",
                        "Como usar exemplos na redação?",
                        "O que é um texto coeso e coerente?",
                        "Como fazer uma proposta de intervenção detalhada?"
                    ]
                )
            ]
        )
        