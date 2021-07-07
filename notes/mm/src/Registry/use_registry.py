from converters.builder import CONVERTERS

converter_cfg = dict(type='Converter1', a=1, b=1)
converter = CONVERTERS.build(converter_cfg)
print(converter)