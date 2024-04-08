import meter.modules.convnext as convnext
convnexts = getattr(convnext, 'convnext_base')(
    pretrained=True
)
print(convnexts)


convnexts2 = getattr(convnext, 'convnext_base')(
    pretrained=True, in_22k = True, num_classes=21841

)
print(convnexts2)