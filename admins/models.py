from django.db import models

class TextCNNMetrics(models.Model):
    accuracy = models.FloatField()
    auc = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set the field to now when the object is created

    def __str__(self):
        return f'Accuracy: {self.accuracy:.2f}, AUC: {self.auc:.2f}, Created at: {self.created_at}'

    class Meta:
        ordering = ['-created_at']  # Order by the most recently created metric first
