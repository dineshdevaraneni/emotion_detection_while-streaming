# Generated by Django 4.0.1 on 2022-02-02 06:24

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='register_model',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('usrname', models.CharField(blank=True, max_length=30, null=True)),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
                ('phone_number', models.IntegerField(blank=True, null=True)),
                ('age', models.IntegerField(blank=True, null=True)),
                ('email_confirm', models.BooleanField(blank=True, default=False, null=True)),
                ('email_code', models.IntegerField(blank=True, null=True)),
                ('message_code', models.IntegerField(blank=True, null=True)),
                ('premium_pass', models.BooleanField(blank=True, default=False, null=True)),
                ('vip_pass', models.BooleanField(blank=True, default=False, null=True)),
                ('classic_pass', models.BooleanField(blank=True, default=True, null=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
